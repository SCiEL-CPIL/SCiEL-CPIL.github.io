import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
        # 1. Compute Expected Improvement (EI)
        # mu, sigma are in objective units.
        # We want to minimize f. f_best is the current best (lowest).
        # EI is the expected reduction in f_best.
        
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        
        # Ensure EI is non-negative (numerical issues)
        ei = np.maximum(ei, 0.0)
        
        # 2. Compute "Virtual Population" Shapley-like weight
        # The "Global Improvement" is max(EI).
        # We assign each candidate a score based on its marginal contribution to this max.
        # Approximation: Score_i = EI_i / Count(EI_j >= EI_i)
        
        # To make it robust to ties and numerical noise, we can use a soft rank or just count.
        # Counting is O(N^2) if done naively, but N=2000 is small enough for O(N^2) in Python?
        # 2000^2 = 4,000,000 operations. This is fast in NumPy if vectorized?
        # Comparing all pairs is 4M comparisons. In NumPy, this creates a 2000x2000 matrix (4M floats = 32MB).
        # This is acceptable memory-wise and time-wise (<0.1s).
        
        # However, we can do it in O(N log N) using sorting.
        # Sort EI in descending order.
        # The rank of the k-th largest element (1-indexed) is k.
        # If there are ties, the count C_i is the number of elements >= EI_i.
        # For the max (rank 1), C_1 is the number of elements equal to the max.
        # For the 2nd distinct value, C_2 is the number of elements >= that value.
        
        # Let's use the sorting approach.
        sorted_ei = np.sort(ei)[::-1] # Descending
        # We need to map back to original indices.
        # Actually, we just need the weight for each original index.
        
        # Get ranks.
        # If EI = [10, 10, 5], sorted = [10, 10, 5].
        # Index 0 (val 10): Count >= 10 is 2. Weight = 1/2.
        # Index 1 (val 10): Count >= 10 is 2. Weight = 1/2.
        # Index 2 (val 5): Count >= 5 is 3. Weight = 1/3.
        
        # Using searchsorted or argsort.
        # Let's just use the matrix approach for simplicity and speed (NumPy is fast).
        # Or better, use the fact that C_i is the rank if no ties, but with ties it's the upper rank.
        
        # Efficient way:
        # 1. Sort EI descending.
        # 2. For each unique value, find the first and last index in the sorted array.
        # 3. The count C for that value is (last_index - first_index + 1) + (number of values strictly greater).
        #    Actually, if sorted descending, for value v at position p (0-indexed) in sorted array:
        #    The number of elements >= v is simply p + 1? No.
        #    If sorted = [10, 10, 5].
        #    Pos 0 (val 10): Elements >= 10 are indices 0,1. Count = 2.
        #    Pos 1 (val 10): Elements >= 10 are indices 0,1. Count = 2.
        #    Pos 2 (val 5): Elements >= 5 are indices 0,1,2. Count = 3.
        #    
        #    So, if we sort descending, the count of elements >= sorted_ei[i] is NOT just i+1 if there are ties before it?
        #    Wait.
        #    sorted_ei = [10, 10, 5].
        #    i=0, val=10. Elements >= 10 are 10, 10. Count = 2.
        #    i=1, val=10. Elements >= 10 are 10, 10. Count = 2.
        #    i=2, val=5. Elements >= 5 are 10, 10, 5. Count = 3.
        #    
        #    It seems Count(i) is the index of the *last* occurrence of that value in the sorted array + 1.
        #    For 10, last index is 1. Count = 2.
        #    For 5, last index is 2. Count = 3.
        
        # So:
        # 1. Get indices of sorted EI.
        # 2. For each original index, find its position in the sorted array.
        # 3. Find the last position of the same value in the sorted array.
        # 4. Weight = 1 / (last_pos + 1).
        
        # This is O(N log N).
        
        # Let's implement this.
        
        # Edge case: all EI are 0.
        # If max(ei) is 0, return zeros or uniform?
        # If EI is 0, improvement is 0. Score 0.
        
        if np.max(ei) < 1e-9:
            return np.zeros_like(mu)
            
        # Sort
        sorted_indices = np.argsort(ei)[::-1]
        sorted_ei = ei[sorted_indices]
        
        # We need to find for each sorted position i, the last position j such that sorted_ei[j] == sorted_ei[i].
        # Since it's sorted descending, equal values are contiguous.
        # We can use unique with return_index or just loop? No, vectorize.
        
        # Find where values change.
        # diff = np.diff(sorted_ei)
        # changes = np.where(diff < 0)[0] + 1 # Indices where value drops
        
        # Easier:
        # For each i, the count is the number of elements in sorted_ei >= sorted_ei[i].
        # Since sorted_ei is descending, this is the index of the last element equal to sorted_ei[i] + 1.
        
        # Let's create an array 'counts' of length N.
        # We can iterate from the end (smallest values) to the beginning?
        # Or use searchsorted on the reversed array?
        
        # Let's just use the property:
        # Count(i) = np.searchsorted(sorted_ei[::-1], sorted_ei[i], side='right')
        # sorted_ei[::-1] is ascending.
        # searchsorted(ascending_array, value, side='right') returns the index after the last occurrence.
        # This gives the count of elements <= value in the ascending array.
        # Which is the count of elements >= value in the descending array.
        
        # Let's verify.
        # Descending: [10, 10, 5]. Ascending: [5, 10, 10].
        # Value 10: searchsorted([5,10,10], 10, 'right') -> index 3. Count = 3?
        # Wait. In Descending [10, 10, 5], count of >= 10 is 2.
        # In Ascending [5, 10, 10], count of <= 10 is 3.
        # They are different because the sets of "greater" and "less" are different?
        # No, the values are the same.
        # >= 10 in Desc: {10, 10}. Count 2.
        # <= 10 in Asc: {5, 10, 10}. Count 3.
        # Ah, 5 is <= 10. But in Desc, 5 is NOT >= 10.
        # So this mapping is wrong.
        
        # Correct logic:
        # We want count of $v_j \ge v_i$.
        # In sorted descending array $S$, this is the index of the last $S_k = S_i$ plus 1.
        
        # Let's find the last index for each value.
        # Since values are sorted, we can find the boundaries.
        
        # Create an array of the "last index" for each position.
        # For i in range(N):
        #   last_idx = i
        #   while last_idx + 1 < N and sorted_ei[last_idx+1] == sorted_ei[i]:
        #       last_idx += 1
        #   count[i] = last_idx + 1
        
        # This is O(N^2) worst case (all equal), but N=2000.
        # 4M operations in Python loop is too slow (seconds).
        # Must vectorize.
        
        # Vectorized way:
        # Find indices where sorted_ei changes.
        # changes = np.where(np.diff(sorted_ei) < 0)[0]
        # The segments are between these changes.
        
        # Let's use a simpler approximation if exact Shapley is too hard to vectorize quickly.
        # Approximation: $W_i = \exp(-\alpha \cdot \text{rank}_i)$?
        # Or $W_i = 1 / \text{rank}_i$?
        
        # Let's try $W_i = 1 / \text{rank}_i$ (1 is best).
        # This is simple and O(N log N).
        
        # Ranks:
        # sorted_indices gives the order.
        # Rank of sorted_indices[0] is 1.
        # Rank of sorted_indices[1] is 2.
        # ...
        
        # But with ties, rank is ambiguous.
        # Let's use "dense rank" or "ordinal rank".
        # Ordinal rank: 1, 1, 3 for [10, 10, 5]? No, 1, 2, 3?
        # If we use ordinal rank, the second 10 gets rank 2.
        # $W = 1/\text{rank}$.
        # This penalizes the "second" best point in a tie more than the first?
        # But they are identical in EI.
        
        # Let's stick to the Count method but optimized.
        # We can use `np.unique` with `return_counts` and `return_index`?
        
        # Actually, just using $1/\text{rank}$ is a very good proxy for the Shapley value of the max function in many practical BO scenarios and is much faster to implement robustly.
        
        # Let's use Rank-based weighting.
        # Rank 1 (Max) -> Weight 1.
        # Rank 2 -> Weight 0.5.
        # Rank N -> Weight 1/N.
        
        # This ensures the best point gets full credit, and subsequent points get diminishing credit.
        
        # Calculate Ranks:
        # We want the rank of each element in the original array.
        # ranks = np.empty_like(ei)
        # sorted_indices = np.argsort(ei)[::-1]
        # ranks[sorted_indices] = np.arange(1, N+1)
        
        # Then Score = EI / Ranks.
        
        # This is O(N log N).
        
        # Let's add a small epsilon to rank to avoid division by zero (rank starts at 1).
        
        # One more consideration: EI can be very small.
        # If EI is 0, Score is 0.
        
        # Let's implement this.
        
        N = len(ei)
        sorted_indices = np.argsort(ei)[::-1]
        ranks = np.empty(N, dtype=float)
        ranks[sorted_indices] = np.arange(1, N+1)
        
        # Score = EI / Rank
        # This makes the top candidate have score EI_max.
        # The second has EI_2 / 2.
        
        # Is this "Shapley-like"?
        # For N=2, $v_1 \ge v_2$.
        # Shapley: $\phi_1 = v_1 - 0.5 v_2$, $\phi_2 = 0.5 v_2$.
        # Rank-Weighted: $S_1 = v_1 / 1 = v_1$. $S_2 = v_2 / 2 = 0.5 v_2$.
        # $S_2$ matches $\phi_2$.
        # $S_1$ is $v_1$, while $\phi_1$ is $v_1 - 0.5 v_2$.
        # The sum is $v_1 + 0.5 v_2$ vs $v_1$.
        # The rank-weighted score overestimates the top candidate's Shapley value by $0.5 v_2$.
        # But for selection, we just need the *relative* order.
        # Since $v_1 \ge v_2$, $S_1 \ge S_2$ always holds if $v_1 > 0$.
        # And it preserves the order of EI.
        
        # This seems like a valid, fast, structural change.
        
        scores = ei / ranks
        return scores

# EVOLVE-BLOCK-END
