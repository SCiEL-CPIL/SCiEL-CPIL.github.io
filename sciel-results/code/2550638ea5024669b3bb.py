import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    def _is_valid(n, adj_bits):
        """Check all subsets of size >= 6 for regularity using bitmasks."""
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                # Compute degrees within the subset using bitmasks
                degrees = []
                for v in comb:
                    deg = 0
                    for w in comb:
                        if v != w and (adj_bits[w] >> v) & 1:
                            deg += 1
                    degrees.append(deg)
                if len(set(degrees)) == 1:
                    return False
        return True

    def _is_valid_fast(n, adj_bits):
        """Faster validity check using numpy for degree computation."""
        # Build adjacency matrix from bitmasks
        adj = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_bits[i] >> j) & 1:
                    adj[i, j] = 1
                    adj[j, i] = 1
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                sub = adj[np.ix_(comb, comb)]
                degs = sub.sum(axis=1)
                if np.all(degs == degs[0]):
                    return False
        return True

    def _count_bad_6(n, adj_bits):
        """Count 6-vertex subsets that induce regular subgraphs."""
        adj = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_bits[i] >> j) & 1:
                    adj[i, j] = 1
                    adj[j, i] = 1
        count = 0
        for comb in itertools.combinations(range(n), 6):
            sub = adj[np.ix_(comb, comb)]
            degs = sub.sum(axis=1)
            if np.all(degs == degs[0]):
                count += 1
        return count

    def _greedy_build(target_n, seed):
        """
        Greedily build a graph on target_n vertices.
        For each new vertex, choose its edges to previous vertices
        to minimize the number of regular 6-sets it creates.
        """
        rng = random.Random(seed)
        adj_bits = [0] * target_n  # adj_bits[i] has bit j set if edge (i,j) exists
        
        for i in range(1, target_n):
            if time.time() > deadline - 5:
                break
            # Previous vertices: 0..i-1
            prev = list(range(i))
            # We need to choose adj_bits[i] (edges to previous vertices)
            # For each 5-subset S of prev, if we add vertex i,
            # the 6-set S ∪ {i} is regular iff:
            #   - all vertices in S have the same degree within S∪{i}
            #   - vertex i has the same degree as them within S∪{i}
            #
            # Strategy: enumerate all 2^i possible edge patterns (if i is small),
            # or use random sampling for larger i.
            
            if i <= 15:
                # Enumerate all 2^i patterns
                best_pattern = 0
                best_score = float('inf')
                all_patterns = range(1 << i)
                # Shuffle for diversity
                pattern_list = list(all_patterns)
                rng.shuffle(pattern_list)
                for pat in pattern_list:
                    # Set adj_bits[i] = pat
                    adj_bits[i] = pat
                    # Count bad 6-sets involving vertex i
                    # A 6-set containing i: pick 5 from prev
                    bad_count = 0
                    for S in itertools.combinations(prev, 5):
                        # Check if S ∪ {i} is regular
                        vertices = list(S) + [i]
                        # Compute degrees
                        degs = []
                        for v in vertices:
                            d = 0
                            for w in vertices:
                                if v == w:
                                    continue
                                if v < i:
                                    # v is in prev, check adj_bits[v] bit w
                                    if w < i:
                                        if (adj_bits[v] >> w) & 1:
                                            d += 1
                                    else:
                                        # w == i
                                        if (pat >> v) & 1:
                                            d += 1
                                else:
                                    # v == i, w in prev
                                    if (pat >> w) & 1:
                                        d += 1
                            degs.append(d)
                        if len(set(degs)) == 1:
                            bad_count += 1
                    if bad_count < best_score:
                        best_score = bad_count
                        best_pattern = pat
                    if bad_count == 0:
                        break
                adj_bits[i] = best_pattern
            else:
                # For larger i, use random sampling + local improvement
                # Start with random pattern
                best_pattern = 0
                for j in range(i):
                    if rng.random() < 0.5:
                        best_pattern |= (1 << j)
                adj_bits[i] = best_pattern
                # Try random flips to improve
                best_score = _count_bad_6(i + 1, adj_bits)
                for _ in range(200):
                    if time.time() > deadline - 5:
                        break
                    # Flip a random edge
                    j = rng.randrange(i)
                    adj_bits[i] ^= (1 << j)
                    new_score = _count_bad_6(i + 1, adj_bits)
                    if new_score <= best_score:
                        best_score = new_score
                    else:
                        adj_bits[i] ^= (1 << j)  # revert
                    if best_score == 0:
                        break

        # Final check
        if time.time() > deadline - 3:
            return None
        
        n_actual = i + 1 if i < target_n else target_n
        # Check validity of the built graph
        if not _is_valid(n_actual, adj_bits):
            return None
        
        # Build edge list
        edges = []
        for u in range(n_actual):
            for v in range(u + 1, n_actual):
                if (adj_bits[u] >> v) & 1:
                    edges.append([u, v])
        return {'n': n_actual, 'edges': edges}

    def _local_search_improve(n, adj_bits, max_iters=500):
        """
        Improve a graph on n vertices via local search.
        Returns improved adj_bits or None if no improvement.
        """
        if time.time() > deadline - 5:
            return None
        
        adj = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_bits[i] >> j) & 1:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        current_bad = _count_bad_6(n, adj_bits)
        if current_bad == 0:
            return adj_bits
        
        best_adj = adj_bits[:]
        best_bad = current_bad
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            u, v = random.choice(edges)
            # Flip edge (u, v)
            adj_bits[u] ^= (1 << v)
            adj_bits[v] ^= (1 << u)
            new_bad = _count_bad_6(n, adj_bits)
            if new_bad <= current_bad:
                current_bad = new_bad
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_adj = adj_bits[:]
                if current_bad == 0:
                    break
            else:
                # Revert
                adj_bits[u] ^= (1 << v)
                adj_bits[v] ^= (1 << u)
        
        if best_bad < current_bad or best_bad == 0:
            return best_adj
        return None

    def _try_target(target_n, time_budget, n_seeds=5):
        """Try to find a valid graph on target_n vertices."""
        t_start = time.time()
        best_result = None
        for seed in range(n_seeds):
            if time.time() > t_start + time_budget:
                break
            result = _greedy_build(target_n, seed)
            if result is not None:
                if result['n'] >= target_n:
                    return result
                # Got a smaller valid graph
                if best_result is None or result['n'] > best_result['n']:
                    best_result = result
            # Try local search improvement on the partial result
            # (already done inside _greedy_build via the final check)
        
        if best_result is not None:
            return best_result
        return None

    # Try to build large graphs
    # Start with n=21 (ambitious), then 20, 19, etc.
    for target in [21, 20, 19, 18]:
        if time.time() > deadline - 10:
            break
        budget = 20 if target >= 20 else 15
        result = _try_target(target, budget, n_seeds=4)
        if result is not None:
            # Try to extend: can we add more vertices?
            # For now, return what we have
            return result

    # Fallback: try smaller
    for target in [17, 16, 15, 12, 10]:
        if time.time() > deadline - 5:
            break
        result = _try_target(target, 8, n_seeds=3)
        if result is not None:
            return result

    # Ultimate fallback: n=6, K6 minus one edge (no regular 6-subset since only one 6-set exists and it's not regular)
    edges = []
    for u in range(6):
        for v in range(u + 1, 6):
            if not (u == 0 and v == 1):
                edges.append([u, v])
    return {'n': 6, 'edges': edges}

