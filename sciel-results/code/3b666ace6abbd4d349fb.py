"""Search for ambient 12-set minimizing largest Sidon subset size."""
import random
import time
from itertools import combinations
from collections import defaultdict

def _is_sidon(S):
    """Check if S is a Sidon set (all pair sums including diagonals are unique)."""
    n = len(S)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _pair_sum_conflicts(A):
    """Count number of pair-sum collisions in A. Higher = more constrained."""
    n = len(A)
    sums = defaultdict(int)
    for i in range(n):
        for j in range(i, n):
            s = A[i] + A[j]
            sums[s] += 1
    # Count collisions: for each sum with count c, there are c-1 "extra" collisions
    collisions = 0
    for c in sums.values():
        if c > 1:
            collisions += c - 1
    return collisions

def _max_sidon_size(A, time_limit=None, deadline=None):
    """Find the maximum size of a Sidon subset of A. Exact for small A using branch & bound."""
    n = len(A)
    if n == 0:
        return 0
    if n == 1:
        return 1
    if _is_sidon(A):
        return n
    
    # Branch and bound: try to build largest Sidon subset
    # Sort A to help with pruning
    A_sorted = sorted(A)
    
    best = [0]
    
    def _branch(start_idx, current, seen_sums, depth):
        if deadline is not None and time.time() > deadline:
            return
        if depth > best[0]:
            best[0] = depth
        # Prune: if remaining elements can't beat best, stop
        if depth + (n - start_idx) <= best[0]:
            return
        for i in range(start_idx, n):
            x = A_sorted[i]
            # Check if x can be added (all pair sums with existing elements must be unique)
            can_add = True
            new_sums = set()
            for j in range(len(current)):
                s = x + current[j]
                if s in seen_sums or s in new_sums:
                    can_add = False
                    break
                new_sums.add(s)
            if can_add:
                s2 = 2 * x
                if s2 in seen_sums or s2 in new_sums:
                    can_add = False
            if can_add:
                new_seen = seen_sums | new_sums | {2 * x}
                _branch(i + 1, current + [x], new_seen, depth + 1)
    
    _branch(0, [], set(), 0)
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    def _validate(A):
        if len(A) != 12:
            return False
        if len(set(A)) != 12:
            return False
        if any(x < 0 or x > 128 for x in A):
            return False
        return True

    best_set = None
    best_size = 12  # We want to minimize this

    def _try_set(A):
        nonlocal best_set, best_size
        if not _validate(A):
            return False
        ms = _max_sidon_size(A, deadline=deadline)
        if ms < best_size:
            best_size = ms
            best_set = A
            # If we found size 5 or less, that's likely optimal, stop early
            if best_size <= 5:
                return True
        return False

    def _try_set_with_prefilter(A, conflict_threshold=20):
        """Quick prefilter: skip sets with very few pair-sum conflicts."""
        nonlocal best_set, best_size
        if not _validate(A):
            return False
        # Sets with many collisions are more likely to have small Sidon subsets
        conflicts = _pair_sum_conflicts(A)
        if conflicts < conflict_threshold:
            return False
        ms = _max_sidon_size(A, deadline=deadline)
        if ms < best_size:
            best_size = ms
            best_set = A
            if best_size <= 5:
                return True
        return False

    # Strategy 1: Exhaustive search over small ranges
    # For range [0, R], number of 12-subsets is C(R+1, 12)
    # R=12: C(13,12)=13
    # R=15: C(16,12)=1820
    # R=18: C(19,12)=50388
    # R=20: C(21,12)=293930
    # R=22: C(23,12)=1352078
    
    for max_val in [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
        if time.time() > deadline - 15.0:
            break
        count = 0
        for subset in combinations(range(max_val + 1), 12):
            A = list(subset)
            _try_set(A)
            count += 1
            if count % 10000 == 0 and time.time() > deadline - 15.0:
                break
        if best_size <= 5:
            break

    # Strategy 2: Structured sets - triples at regular intervals
    for gap in range(3, 30):
        for offset in range(0, 129 - 3*gap - 2):
            if time.time() > deadline - 15.0:
                break
            A = []
            for k in range(4):
                base = offset + k * gap
                if base + 2 > 128:
                    break
                A.extend([base, base+1, base+2])
            if len(A) == 12 and len(set(A)) == 12:
                _try_set(A)
        if best_size <= 5:
            break

    # Strategy 3: Pairs at regular intervals
    for gap in range(2, 25):
        for offset in range(0, 129 - 6*gap - 1):
            if time.time() > deadline - 15.0:
                break
            A = []
            for k in range(6):
                base = offset + k * gap
                if base + 1 > 128:
                    break
                A.extend([base, base+1])
            if len(A) == 12 and len(set(A)) == 12:
                _try_set(A)
        if best_size <= 5:
            break

    # Strategy 4: Quadruples at regular intervals
    for gap in range(4, 20):
        for offset in range(0, 129 - 4*gap - 3):
            if time.time() > deadline - 15.0:
                break
            A = []
            for k in range(3):
                base = offset + k * gap
                if base + 3 > 128:
                    break
                A.extend([base, base+1, base+2, base+3])
            if len(A) == 12 and len(set(A)) == 12:
                _try_set(A)
        if best_size <= 5:
            break

    # Strategy 5: Dense clusters - consecutive blocks
    # Try various configurations of consecutive number blocks
    # e.g., [0,1,2,3], [5,6,7], [9,10,11,12], [14,15,16]
    for n_blocks in range(2, 7):
        if time.time() > deadline - 15.0:
            break
        # Generate all ways to partition 12 into n_blocks consecutive blocks
        def _gen_block_sizes(total, n_parts, min_size=1):
            if n_parts == 1:
                if total >= min_size:
                    yield (total,)
                return
            for i in range(min_size, total - (n_parts - 1) * min_size + 1):
                for rest in _gen_block_sizes(total - i, n_parts - 1, min_size):
                    yield (i,) + rest
        
        for sizes in _gen_block_sizes(12, n_blocks):
            # Try different gap patterns between blocks
            max_gap = max(2, (128 - sum(sizes)) // max(n_blocks - 1, 1))
            for gap in range(1, min(max_gap, 15) + 1):
                if time.time() > deadline - 15.0:
                    break
                for offset in range(0, 30):
                    A = []
                    pos = offset
                    valid = True
                    for s in sizes:
                        for v in range(s):
                            if pos + v > 128:
                                valid = False
                                break
                            A.append(pos + v)
                        if not valid:
                            break
                        pos += s + gap
                    if valid and len(A) == 12 and len(set(A)) == 12:
                        _try_set(A)
                    if time.time() > deadline - 15.0:
                        break
            if best_size <= 5:
                break
        if best_size <= 5:
            break

    # Strategy 6: Random search biased toward dense clusters
    rng = random.Random(42)
    for _ in range(50000):
        if time.time() > deadline - 20.0:
            break
        # Generate dense cluster: pick a few base points, add neighbors
        n_clusters = rng.randint(2, 5)
        A = set()
        for _ in range(n_clusters):
            base = rng.randint(0, 120)
            cluster_size = rng.randint(2, 5)
            for j in range(cluster_size):
                val = base + j
                if 0 <= val <= 128:
                    A.add(val)
            if len(A) >= 12:
                break
        # Fill remaining with random values
        while len(A) < 12:
            A.add(rng.randint(0, 128))
        A = sorted(list(A)[:12])
        _try_set(A)
        if best_size <= 5:
            break

    # Strategy 7: Local search - perturb best found set with multiple mutations
    if best_set is not None:
        for _ in range(25000):
            if time.time() > deadline - 5.0:
                break
            A = list(best_set)
            # Randomly swap one or two elements
            num_mutations = rng.choice([1, 1, 2, 2, 3])
            for _ in range(num_mutations):
                idx = rng.randint(0, 11)
                old_val = A[idx]
                # Bias toward values close to existing ones (dense clustering)
                if rng.random() < 0.6:
                    # Pick a value near an existing element
                    ref_idx = rng.randint(0, 11)
                    ref_val = A[ref_idx]
                    new_val = ref_val + rng.randint(-3, 3)
                    if new_val < 0:
                        new_val = 0
                    elif new_val > 128:
                        new_val = 128
                else:
                    new_val = rng.randint(0, 128)
                if new_val != old_val and new_val not in A:
                    A[idx] = new_val
            A.sort()
            _try_set(A)
            if best_size <= 5:
                break

    # Strategy 8: Structured - arithmetic progressions with perturbations
    for base_gap in range(4, 20):
        for _ in range(800):
            if time.time() > deadline - 5.0:
                break
            A = []
            offset = rng.randint(0, 20)
            for i in range(12):
                val = offset + i * base_gap
                # Add small perturbation
                pert = rng.choice([-2, -1, 0, 0, 0, 1, 2])
                val += pert
                if 0 <= val <= 128 and val not in A:
                    A.append(val)
            if len(A) == 12:
                A.sort()
                _try_set(A)
            if best_size <= 5:
                break
        if best_size <= 5:
            break

    # Strategy 9: Pairs with varying gaps
    for _ in range(8000):
        if time.time() > deadline - 3.0:
            break
        A = []
        offset = rng.randint(0, 10)
        for i in range(6):
            gap = rng.randint(3, 20)
            base = offset + i * gap
            if base + 1 > 128:
                break
            A.extend([base, base + 1])
        if len(A) == 12:
            A.sort()
            _try_set(A)
        if best_size <= 5:
            break

    # Strategy 10: Final polish - focused local search around best
    if best_set is not None and best_size > 5:
        for _ in range(20000):
            if time.time() > deadline - 2.0:
                break
            A = list(best_set)
            # Single mutation with dense bias
            idx = rng.randint(0, 11)
            old_val = A[idx]
            ref_idx = rng.randint(0, 11)
            ref_val = A[ref_idx]
            new_val = ref_val + rng.randint(-4, 4)
            new_val = max(0, min(128, new_val))
            if new_val != old_val and new_val not in A:
                A[idx] = new_val
            A.sort()
            _try_set(A)
            if best_size <= 5:
                break

    # Fallback
    if best_set is None:
        best_set = [0,1,2, 10,11,12, 20,21,22, 30,31,32]
    
    if not _validate(best_set):
        best_set = [3*i for i in range(12)]
    
    return sorted(best_set)
