"""Rewritten program using randomized local search with NumPy acceleration."""
import numpy as np

def construct():
    """
    Construct a B2[2] set (every sum has at most 2 representations a+b with a<=b)
    from {1,...,1000}, maximizing size.
    
    Strategy:
    1. Start with a greedy baseline to get a valid set.
    2. Use randomized local search: start from a dense random set, then iteratively
       remove/swap elements to fix violations while trying to maintain/increase size.
    3. Use NumPy for efficient sum-count computation.
    """
    N = 1000
    max_sum = 2 * N  # max sum is 2000
    
    def compute_counts(chosen_set):
        """Compute sum representation counts using boolean array convolution."""
        if not chosen_set:
            return np.zeros(max_sum + 1, dtype=np.int32)
        arr = np.zeros(N + 1, dtype=np.int32)
        for x in chosen_set:
            arr[x] = 1
        # Convolution gives count of pairs (i,j) with i+j = s, i<=j, both in set
        # np.convolve with 'full' mode; we want sums from 2 to 2000
        conv = np.convolve(arr, arr, mode='full')
        # conv[k] = sum_{i+j=k} arr[i]*arr[j] for i,j in 0..N
        # We need to handle a<=b constraint:
        # For s = i+j with i<j: counted twice in conv (once for (i,j) and (j,i))
        # For s = 2i: counted once
        # We want count of pairs (a,b) with a<=b, a+b=s, both in set
        # conv[s] = (# pairs with a<b) * 2 + (# pairs with a=b)
        # Let c_ab = # pairs a<b with a+b=s, c_aa = # pairs a=a with 2a=s
        # conv[s] = 2*c_ab + c_aa
        # We want c_ab + c_aa <= 2
        # Note: c_aa is 1 if s is even and s/2 in set, else 0
        # So: c_ab + c_aa = (conv[s] - c_aa) / 2 + c_aa = (conv[s] + c_aa) / 2
        # Let's compute directly:
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        for s in range(2, max_sum + 1):
            c_aa = 1 if (s % 2 == 0 and s // 2 <= N and arr[s // 2] == 1) else 0
            c_ab = (conv[s] - c_aa) // 2 if conv[s] >= c_aa else 0
            counts[s] = c_ab + c_aa
        return counts
    
    def is_valid(chosen_set):
        """Check if chosen_set is a valid B2[2] set."""
        if not chosen_set:
            return True
        counts = compute_counts(chosen_set)
        return np.max(counts) <= 2
    
    def greedy_construct():
        """Greedy construction to get a valid baseline."""
        chosen = []
        counts = {}
        for x in range(1, N + 1):
            new_sums = [x + y for y in chosen] + [2 * x]
            trial = dict(counts)
            valid = True
            for s in new_sums:
                trial[s] = trial.get(s, 0) + 1
                if trial[s] > 2:
                    valid = False
                    break
            if valid:
                chosen.append(x)
                counts = trial
        return chosen
    
    # Step 1: Get a greedy baseline
    baseline = greedy_construct()
    best_set = set(baseline)
    best_size = len(baseline)
    
    # Step 2: Randomized local search
    # Start from a dense random initialization
    np.random.seed(42)
    
    # Try multiple random restarts
    for restart in range(50):
        # Initialize with a random subset of size around 0.7*N
        target_size = int(0.7 * N)
        candidates = np.random.choice(N, size=target_size, replace=False) + 1
        current_set = set(candidates.tolist())
        
        # Fix violations by removing elements
        # Iteratively remove elements that cause the most violations
        for iteration in range(200):
            if not current_set:
                break
            
            counts = compute_counts(current_set)
            max_count = np.max(counts)
            
            if max_count <= 2:
                # Valid set! Update best if larger
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
                break
            
            # Find elements involved in violations
            # For each sum s with count > 2, find pairs that sum to s
            violating_sums = np.where(counts > 2)[0]
            
            # Score each element by how many violations it's involved in
            element_score = {x: 0 for x in current_set}
            for s in violating_sums:
                # Find all pairs (a,b) in current_set with a<=b and a+b=s
                for a in current_set:
                    b = s - a
                    if b < a:
                        continue
                    if b in current_set and b <= N:
                        element_score[a] += 1
                        if b != a:
                            element_score[b] += 1
            
            # Remove the element with highest score
            if element_score:
                worst = max(element_score, key=element_score.get)
                current_set.discard(worst)
        
        # Try to add back some elements
        if len(current_set) < best_size:
            # Try adding random elements
            for _ in range(100):
                if not current_set:
                    break
                counts = compute_counts(current_set)
                if np.max(counts) <= 2:
                    # Try to add a random element
                    available = [x for x in range(1, N + 1) if x not in current_set]
                    if not available:
                        break
                    x = np.random.choice(available)
                    # Check if adding x keeps validity
                    new_sums = [x + y for y in current_set] + [2 * x]
                    valid = True
                    for s in new_sums:
                        if counts[s] + 1 > 2:
                            valid = False
                            break
                    if valid:
                        current_set.add(x)
                else:
                    # Still has violations, remove one
                    violating_sums = np.where(counts > 2)[0]
                    if violating_sums:
                        s = violating_sums[0]
                        for a in current_set:
                            b = s - a
                            if b >= a and b in current_set:
                                # Remove one of them
                                if a < b:
                                    current_set.discard(a)
                                else:
                                    current_set.discard(b)
                                break
    
    # Step 3: Local search from best set found
    # Try swapping elements to increase size
    for _ in range(100):
        if len(best_set) >= N:
            break
        counts = compute_counts(best_set)
        if np.max(counts) > 2:
            # Fix violations first
            violating_sums = np.where(counts > 2)[0]
            for s in violating_sums[:5]:
                for a in list(best_set):
                    b = s - a
                    if b >= a and b in best_set:
                        best_set.discard(a)
                        break
            continue
        
        # Try adding a random element
        available = [x for x in range(1, N + 1) if x not in best_set]
        if not available:
            break
        x = np.random.choice(available)
        new_sums = [x + y for y in best_set] + [2 * x]
        valid = True
        for s in new_sums:
            if counts[s] + 1 > 2:
                valid = False
                break
        if valid:
            best_set.add(x)
        else:
            # Try swapping: remove some element y and add x
            # Find y that when removed, allows x to be added
            for y in list(best_set):
                if y == x:
                    continue
                # Check if removing y and adding x is valid
                # New sums with x: x + z for z in (best_set - {y}) plus 2x
                # Removed sums: y + z for z in (best_set - {y}) plus 2y
                # This is complex, so just try a few random swaps
                trial_set = set(best_set)
                trial_set.discard(y)
                trial_set.add(x)
                trial_counts = compute_counts(trial_set)
                if np.max(trial_counts) <= 2:
                    best_set = trial_set
                    break
    
    return sorted(best_set)

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    # Verify
    counts = {}
    for i, a in enumerate(result):
        for b in result[i:]:
            s = a + b
            counts[s] = counts.get(s, 0) + 1
    max_rep = max(counts.values()) if counts else 0
    print(f"Max representations: {max_rep}")
    print(f"Valid: {max_rep <= 2}")
