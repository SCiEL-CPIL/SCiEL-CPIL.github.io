"""Rewritten program with a stronger algorithm for integer sum-product extremal sets."""
import time
import random
import sys
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 75.0  # Safety margin before 90s kill
    best_set = None
    best_score = float('inf')
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using numpy for speed."""
        A = np.array(sorted(A), dtype=np.int64)
        n = len(A)
        # Compute all pairwise sums (i <= j)
        i_idx, j_idx = np.triu_indices(n)
        sums = A[i_idx] + A[j_idx]
        sumset_size = len(np.unique(sums))
        # Compute all pairwise products (i <= j)
        prods = A[i_idx] * A[j_idx]
        prodset_size = len(np.unique(prods))
        return max(sumset_size, prodset_size)
    
    def generate_smooth_numbers(limit=10000):
        """Generate all 7-smooth numbers <= limit."""
        nums = set([1])
        # Use a max-heap or just generate all combinations
        # 2^a * 3^b * 5^c * 7^d <= 10000
        # Max a: log2(10000) ~ 13
        # Max b: log3(10000) ~ 8
        # Max c: log5(10000) ~ 5
        # Max d: log7(10000) ~ 4
        for a in range(14):
            p2 = 2 ** a
            if p2 > limit: break
            for b in range(9):
                p3 = 3 ** b
                if p2 * p3 > limit: break
                for c in range(6):
                    p5 = 5 ** c
                    if p2 * p3 * p5 > limit: break
                    for d in range(5):
                        p7 = 7 ** d
                        val = p2 * p3 * p5 * p7
                        if val <= limit:
                            nums.add(val)
        return sorted(nums)

    smooth_numbers = generate_smooth_numbers(10000)
    # Precompute indices for fast lookup if needed, but we'll just pick from the list
    
    random.seed(42)
    
    # Initial candidates: Subsets of smooth numbers
    # Strategy 1: Dense small smooth numbers
    # Strategy 2: Mixed small and large smooth numbers
    # Strategy 3: Arithmetic-like progressions within smooth numbers (harder, just pick random subsets)
    
    candidates = []
    
    # Take the first 20 smooth numbers
    if len(smooth_numbers) >= 20:
        candidates.append(smooth_numbers[:20])
        
    # Take smooth numbers spread out
    if len(smooth_numbers) >= 20:
        # Pick indices that are spaced out
        indices = np.linspace(0, len(smooth_numbers)-1, 20).astype(int)
        candidates.append([smooth_numbers[i] for i in indices])
        
    # Random subsets of smooth numbers
    for _ in range(20):
        subset = random.sample(smooth_numbers, 20)
        candidates.append(sorted(subset))
        
    # Also keep the original sequential candidate as a fallback
    candidates.append(list(range(1, 21)))
    
    # Evaluate initial candidates
    for cand in candidates:
        if len(cand) != 20 or len(set(cand)) != 20 or min(cand) < 1 or max(cand) > 10000:
            continue
        score = evaluate(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
            
    if best_set is None:
        best_set = list(range(1, 21))
        best_score = evaluate(best_set)
        
    # Local Search with Incremental Updates
    # We maintain counts of sums and products to update quickly.
    # State: A (sorted list of 20 ints)
    # Data structures: 
    #   sum_counts: Counter of all sums a_i + a_j (i<=j)
    #   prod_counts: Counter of all products a_i * a_j (i<=j)
    
    def build_counters(A):
        A = sorted(A)
        n = len(A)
        sum_counts = Counter()
        prod_counts = Counter()
        for i in range(n):
            for j in range(i, n):
                s = A[i] + A[j]
                p = A[i] * A[j]
                sum_counts[s] += 1
                prod_counts[p] += 1
        return sum_counts, prod_counts
        
    def get_score_from_counters(sum_counts, prod_counts):
        # Size of set is number of keys with count > 0
        return max(len(sum_counts), len(prod_counts))
        
    current = best_set[:]
    current.sort()
    sum_counts, prod_counts = build_counters(current)
    current_score = get_score_from_counters(sum_counts, prod_counts)
    if current_score < best_score:
        best_score = current_score
        best_set = current[:]
        
    # Precompute neighbors for smooth numbers? No, just sample from smooth_numbers.
    # To speed up, we can restrict new values to smooth numbers.
    
    temp = 10.0
    cooling = 0.9995
    
    # Time budget check
    # Each step involves:
    # 1. Removing 2n-1 sums and 2n-1 prods from counters (O(n))
    # 2. Adding 2n-1 sums and 2n-1 prods to counters (O(n))
    # 3. Checking size of counters (O(1) if we track size, but len() is O(1) in Python for dict/Counter? 
    #    Actually, len(Counter) is O(1) as it's just len(dict). 
    #    Wait, we need to know if a key becomes 0 to remove it? 
    #    Counter in Python doesn't auto-remove zero keys, but len(Counter) counts keys with value > 0?
    #    No, len(Counter) returns the number of keys with non-zero counts? 
    #    Let's check: Counter({'a': 0}) -> len is 1? No, Counter treats 0 as present?
    #    Actually, `len(Counter({'a': 0}))` is 1. But we want the size of the *set* of sums.
    #    So if a sum count drops to 0, it should no longer contribute to the set size.
    #    Standard Counter keeps the key. We need to filter or use a different structure.
    #    Using a dict and manually deleting keys when count hits 0 is safer for len().
    
    # Let's switch to standard dict for counters to ensure len() reflects unique values with count > 0.
    
    def build_counters_dict(A):
        A = sorted(A)
        n = len(A)
        sum_counts = {}
        prod_counts = {}
        for i in range(n):
            for j in range(i, n):
                s = A[i] + A[j]
                p = A[i] * A[j]
                sum_counts[s] = sum_counts.get(s, 0) + 1
                prod_counts[p] = prod_counts.get(p, 0) + 1
        return sum_counts, prod_counts
        
    def get_score_dict(sum_counts, prod_counts):
        return max(len(sum_counts), len(prod_counts))
        
    sum_counts, prod_counts = build_counters_dict(current)
    current_score = get_score_dict(sum_counts, prod_counts)
    
    # Optimization: Pre-compute the "impact" of removing/adding an element?
    # When we replace A[i] with X:
    # 1. Remove contributions of A[i]:
    #    For all j != i: remove (A[i]+A[j]) from sum_counts, (A[i]*A[j]) from prod_counts
    #    Remove (A[i]+A[i]) from sum_counts, (A[i]*A[i]) from prod_counts
    # 2. Add contributions of X:
    #    For all j != i: add (X+A[j]) to sum_counts, (X*A[j]) to prod_counts
    #    Add (X+X) to sum_counts, (X*X) to prod_counts
    #    Note: The index i is fixed in the sorted list, but the value changes.
    #    Since we maintain a sorted list, inserting X might change the order.
    #    However, the set of pairs (i,j) with i<j is defined by indices.
    #    If we just swap the value at index i, the set of indices involved in sums/prods remains the same.
    #    The values change.
    
    def update_counters_remove(sum_counts, prod_counts, A, idx):
        val = A[idx]
        n = len(A)
        # Remove val + A[j] for all j != idx
        for j in range(n):
            if j == idx:
                continue
            s = val + A[j]
            p = val * A[j]
            
            sum_counts[s] -= 1
            if sum_counts[s] == 0:
                del sum_counts[s]
            prod_counts[p] -= 1
            if prod_counts[p] == 0:
                del prod_counts[p]
        
        # Remove val + val (diagonal)
        s = val + val
        p = val * val
        sum_counts[s] -= 1
        if sum_counts[s] == 0:
            del sum_counts[s]
        prod_counts[p] -= 1
        if prod_counts[p] == 0:
            del prod_counts[p]
            
    def update_counters_add(sum_counts, prod_counts, A, idx, new_val):
        n = len(A)
        # Add new_val + A[j] for all j != idx
        # Note: A[idx] is currently the old value in the list A? 
        # We should pass the list with the new value at idx, or just use the other values.
        # Let's assume A[idx] is updated to new_val before calling this?
        # Or better, pass the other values.
        # To avoid confusion, let's just iterate over A and skip idx.
        
        for j in range(n):
            if j == idx:
                continue
            s = new_val + A[j]
            p = new_val * A[j]
            
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        # Add new_val + new_val (diagonal)
        s = new_val + new_val
        p = new_val * new_val
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1

    # Simulation Loop
    # We will restrict new values to smooth_numbers to maintain low product set size.
    # We can also allow some random integers if we get stuck, but smooth numbers are a strong prior.
    
    # To make the search faster, we can pre-select a pool of candidate values.
    # Let's use the top 200 smooth numbers as the pool for mutations.
    # If the current set contains large numbers, we might want a larger pool.
    # Let's just use all smooth numbers <= 10000. There are about 200-300 of them.
    # Sampling from this list is O(1).
    
    num_iterations = 500000
    for iteration in range(num_iterations):
        if time.time() > deadline:
            break
        
        if temp < 0.1:
            temp = 5.0
            # Restart from best found so far
            current = best_set[:]
            current.sort()
            sum_counts, prod_counts = build_counters_dict(current)
            current_score = get_score_dict(sum_counts, prod_counts)
            if current_score < best_score:
                best_score = current_score
                best_set = current[:]
                
        # Pick a random index to modify
        i = random.randint(0, 19)
        old_val = current[i]
        
        # Pick a new value from smooth numbers
        # To encourage diversity, pick from the pool
        new_val = random.choice(smooth_numbers)
        
        # If new_val is same as old_val, skip
        if new_val == old_val:
            continue
            
        # Check for duplicates in current set
        # Since we are replacing current[i], we just need to ensure new_val is not in current[j] for j != i
        # This is O(n) check.
        is_dup = False
        for j in range(20):
            if j != i and current[j] == new_val:
                is_dup = True
                break
        if is_dup:
            continue
            
        # Perform the swap
        # 1. Remove old_val contributions
        update_counters_remove(sum_counts, prod_counts, current, i)
        
        # 2. Update current list
        current[i] = new_val
        # Note: We don't re-sort immediately for the counter update logic above, 
        # because the indices i and j are fixed. The set of pairs is determined by indices.
        # However, the values A[j] used in the loop must be the *current* values of the other elements.
        # Since we only changed A[i], the other A[j] are unchanged.
        # So the logic holds.
        
        # 3. Add new_val contributions
        update_counters_add(sum_counts, prod_counts, current, i, new_val)
        
        # 4. Calculate new score
        new_score = get_score_dict(sum_counts, prod_counts)
        delta_score = new_score - current_score
        
        # Acceptance criterion
        if delta_score < 0 or (delta_score == 0 and random.random() < 0.5) or \
           (delta_score > 0 and random.random() < np.exp(-delta_score / temp)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current[:]
        else:
            # Reject: revert
            # 1. Remove new_val contributions
            update_counters_remove(sum_counts, prod_counts, current, i)
            
            # 2. Revert list
            current[i] = old_val
            
            # 3. Add old_val contributions back
            update_counters_add(sum_counts, prod_counts, current, i, old_val)
            
        temp *= cooling
        
    # Final polish: Try to sort and maybe do a few more random swaps from the full integer range?
    # The smooth number restriction is strong. Let's stick with it.
    
    return best_set
# EVOLVE-BLOCK-END
