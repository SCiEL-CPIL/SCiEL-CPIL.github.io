"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter
from itertools import combinations

def _count_duplicate_sums(subset):
    """Count how many unordered pair sums (including diagonals) are duplicated."""
    n = len(subset)
    sums = [subset[i] + subset[j] for i in range(n) for j in range(i, n)]
    c = Counter(sums)
    return sum(k * (k - 1) // 2 for k in c.values())

def _is_sidon(vals):
    """Check if a list has all unordered pair sums unique (including diagonals)."""
    s = len(vals)
    seen = set()
    for i in range(s):
        for j in range(i, s):
            ps = vals[i] + vals[j]
            if ps in seen:
                return False
            seen.add(ps)
    return True

def _max_sidon_subset_size_fast(subset, deadline=None):
    """
    Find largest subset with all unordered pair sums unique (including diagonals).
    Uses greedy upper bound + targeted search.
    """
    n = len(subset)
    if n <= 1:
        return n
    
    # Quick upper bound: if total pairs > possible unique sums in range
    # Sum range for 12 elements in [0,128]: min sum=0, max sum=256, so 257 possible sums
    # Number of pairs including diagonals: n*(n+1)/2
    # If n*(n+1)/2 > 257, then by pigeonhole, not all can be unique
    # For n=12: 78 pairs, which is < 257, so this bound doesn't help directly
    
    # Try to find a large Sidon subset
    # Start with greedy: add elements that don't create conflicts
    def greedy_sidon_size():
        remaining = list(range(n))
        chosen = []
        while remaining:
            best_idx = None
            best_conflicts = float('inf')
            for idx in remaining:
                # Count conflicts if we add this element
                conflicts = 0
                val = subset[idx]
                for j in range(len(chosen)):
                    # Check if val + subset[chosen[j]] conflicts with existing sums
                    # We need to track existing pair sums
                    pass
                # Simpler: just try adding and check
                test_chosen = chosen + [idx]
                if _is_sidon([subset[i] for i in test_chosen]):
                    best_idx = idx
                    best_conflicts = 0
                    break
            if best_idx is None:
                break
            chosen.append(best_idx)
            remaining.remove(best_idx)
        return len(chosen)
    
    greedy_size = greedy_sidon_size()
    
    # If greedy found size 12, return 12
    if greedy_size == n:
        return n
    
    # Try to find if we can do better than greedy with targeted search
    # Use backtracking with pruning for sizes from greedy_size+1 up
    # But limit the search time
    
    # For each candidate size from n down to greedy_size+1, try to find a Sidon subset
    # This is expensive, so we limit it
    
    # Alternative: use the fact that for small n, we can try all subsets of size k
    # But C(12, 11) = 12, C(12, 10) = 66, C(12, 9) = 220, C(12, 8) = 495
    
    # Let's try sizes from n down to greedy_size+1, but with time limit
    for size in range(n, greedy_size, -1):
        if deadline and time.time() > deadline:
            break
        # Try all combinations of this size
        found = False
        count = 0
        for combo in combinations(range(n), size):
            if deadline and time.time() > deadline:
                break
            count += 1
            vals = [subset[i] for i in combo]
            if _is_sidon(vals):
                found = True
                break
            # Limit the number of combinations to check per size
            if count > 2000:
                break
        if found:
            return size
    
    return greedy_size

def _local_search_maximize_collisions(subset, deadline, max_passes=20, temp_start=1.0, temp_end=0.01):
    """
    Simulated annealing to maximize collision count (duplicate pair sums).
    """
    subset = list(subset)
    n = len(subset)
    current_dups = _count_duplicate_sums(subset)
    best_dups = current_dups
    best_subset = list(subset)
    
    for p in range(max_passes):
        if time.time() > deadline:
            break
        
        # Temperature schedule
        progress = p / max_passes
        temp = temp_start * (temp_end / temp_start) ** progress
        
        improved_this_pass = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            old_val = subset[i]
            # Try a random new value
            for _ in range(50):  # Try multiple random values per position
                if time.time() > deadline:
                    break
                
                new_val = random.randint(0, 128)
                if new_val == old_val:
                    continue
                # Check uniqueness
                if any(subset[j] == new_val for j in range(n) if j != i):
                    continue
                
                subset[i] = new_val
                new_dups = _count_duplicate_sums(subset)
                
                delta = new_dups - current_dups
                if delta > 0 or random.random() < (temp ** (1.0 / max(delta, 0.1))):
                    # Accept
                    current_dups = new_dups
                    if new_dups > best_dups:
                        best_dups = new_dups
                        best_subset = list(subset)
                    improved_this_pass = True
                    break
                else:
                    # Reject
                    subset[i] = old_val
            
            if improved_this_pass:
                break
        
        if not improved_this_pass:
            # No improvement, maybe restart with best
            if best_dups > current_dups:
                subset = list(best_subset)
                current_dups = best_dups
    
    return best_subset, best_dups

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    random.seed(42)
    
    best_subset = list(range(12))
    best_score = _max_sidon_subset_size_fast(best_subset, deadline)
    best_dups = _count_duplicate_sums(best_subset)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Random candidates
    for _ in range(30):
        candidates.append(random.sample(range(129), 12))
    
    # 2. Structured: dense clusters
    for center in range(20, 120, 15):
        # Cluster of 12 in a small range
        spread = random.choice([3, 4, 5, 6])
        cluster = []
        for i in range(12):
            val = center + random.randint(-spread, spread)
            val = max(0, min(128, val))
            while val in cluster:
                val = (val + 1) % 129
            cluster.append(val)
        candidates.append(cluster)
    
    # 3. Structured: two dense clusters
    for c1 in range(10, 60, 15):
        for c2 in range(70, 120, 15):
            if c2 - c1 < 20:
                continue
            cluster1 = []
            cluster2 = []
            for i in range(6):
                v = c1 + random.randint(-3, 3)
                v = max(0, min(128, v))
                while v in cluster1 or v in cluster2:
                    v = (v + 1) % 129
                cluster1.append(v)
            for i in range(6):
                v = c2 + random.randint(-3, 3)
                v = max(0, min(128, v))
                while v in cluster1 or v in cluster2:
                    v = (v + 1) % 129
                cluster2.append(v)
            candidates.append(cluster1 + cluster2)
    
    # 4. Arithmetic progressions with small difference
    for start in range(0, 100, 8):
        for d in [1, 2, 3]:
            ap = [start + i * d for i in range(12)]
            if all(0 <= x <= 128 for x in ap):
                candidates.append(ap)
    
    # 5. Geometric-like sequences
    for base in range(2, 6):
        geo = []
        val = 1
        for i in range(12):
            geo.append(val % 129)
            val = (val * base) % 129
        # Ensure distinctness
        if len(set(geo)) == 12:
            candidates.append(geo)
    
    # Process candidates
    for candidate in candidates:
        if time.time() > deadline - 10:
            break
        
        # Ensure 12 distinct elements in [0, 128]
        if len(set(candidate)) != 12 or not all(0 <= x <= 128 for x in candidate):
            continue
        
        # Local search to maximize collisions
        improved_candidate, dups = _local_search_maximize_collisions(candidate, deadline, max_passes=10)
        
        # Evaluate the actual score
        score = _max_sidon_subset_size_fast(improved_candidate, deadline)
        
        if score < best_score:
            best_score = score
            best_subset = improved_candidate
            best_dups = dups
            if best_score <= 5:
                break
        elif dups > best_dups and score == best_score:
            # Same score but more collisions, might be more robust
            best_subset = improved_candidate
            best_dups = dups
    
    # Final polish: try more intensive local search on the best
    if time.time() < deadline - 15 and best_score > 5:
        polished, _ = _local_search_maximize_collisions(best_subset, deadline, max_passes=30, temp_start=2.0, temp_end=0.05)
        polished_score = _max_sidon_subset_size_fast(polished, deadline)
        if polished_score < best_score:
            best_score = polished_score
            best_subset = polished
    
    return sorted(best_subset)
# EVOLVE-BLOCK-END
