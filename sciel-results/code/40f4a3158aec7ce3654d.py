import time
import sys

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s kill
    
    n = 12
    
    def subset_sum_count(seq):
        """Count distinct subset sums using bitset (Python int as bitset)."""
        bits = 1  # only sum 0 achievable initially
        for a in seq:
            bits |= (bits << a)
        return bin(bits).count('1')
    
    def is_valid(seq):
        """Check if all subset sums are distinct."""
        return subset_sum_count(seq) == (1 << len(seq))
    
    def greedy_construct(n=12):
        """Greedy: pick smallest valid element at each step."""
        seq = []
        for i in range(n):
            start = 1 if not seq else seq[-1] + 1
            for a in range(start, 1048577):
                candidate = seq + [a]
                if is_valid(candidate):
                    seq.append(a)
                    break
            else:
                return None
        return seq
    
    def local_search(seq, time_limit):
        """Hill-climbing: try to reduce each element to minimize max."""
        current = list(seq)
        cur_max = max(current)
        best_seq = list(current)
        best_max = cur_max
        start_ls = time.time()
        
        improved = True
        while improved:
            improved = False
            if time.time() - start_ls > time_limit:
                break
            
            for idx in range(n):
                if time.time() - start_ls > time_limit:
                    break
                old_val = current[idx]
                # Try reducing this element
                for new_val in range(1, old_val):
                    if time.time() - start_ls > time_limit:
                        break
                    candidate = list(current)
                    candidate[idx] = new_val
                    # Check distinctness
                    if len(set(candidate)) != n:
                        continue
                    if is_valid(candidate):
                        new_max = max(candidate)
                        if new_max < cur_max or (new_max == cur_max and sum(candidate) < sum(current)):
                            current = candidate
                            cur_max = new_max
                            improved = True
                            break
                if improved:
                    if cur_max < best_max:
                        best_max = cur_max
                        best_seq = list(current)
                    break
        
        return best_seq, best_max
    
    # Start with greedy construction
    seq = greedy_construct(n)
    if seq is None:
        return [1 << i for i in range(n)]
    
    best_seq = list(seq)
    best_max = max(best_seq)
    
    # Local search to improve
    improved_seq, improved_max = local_search(seq, time_limit - (time.time() - start_time) - 5)
    if improved_max < best_max:
        best_seq = improved_seq
        best_max = improved_max
    
    # Try multiple random restarts if time permits
    import random
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit - 5:
        # Perturb the best solution
        base = list(best_seq)
        for _ in range(rng.randint(1, 3)):
            idx = rng.randint(0, n - 1)
            offset = rng.randint(-20, 20)
            new_val = base[idx] + offset
            if new_val >= 1 and new_val <= 1048576:
                base[idx] = new_val
        base.sort()
        
        if len(set(base)) == n and is_valid(base):
            perturbed_seq, perturbed_max = local_search(
                base, min(10.0, time_limit - (time.time() - start_time) - 5)
            )
            if perturbed_max < best_max:
                best_seq = perturbed_seq
                best_max = perturbed_max
    
    return best_seq
