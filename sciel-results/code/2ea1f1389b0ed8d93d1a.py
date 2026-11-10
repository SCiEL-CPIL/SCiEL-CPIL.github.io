import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    N = 16
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    M = len(all_edges)  # 120
    
    # Represent graph as bitmasks for speed
    # adj_mask[u] is a 16-bit integer where bit v is set if edge (u,v) exists
    adj_mask = [(1 << N) - 2 for _ in range(N)]  # all edges initially
    for u in range(N):
        adj_mask[u] &= ~(1 << u)  # no self-loops
    
    def has_q3_fast(masks):
        """Fast check for Q3 subgraph using bitmask operations."""
        # Q3 = K_{4,4} minus a perfect matching
        # For each 4-set A and 4-set B (disjoint), check if
        # the bipartite graph is K_{4,4} minus a perfect matching
        for A in itertools.combinations(range(N), 4):
            A_mask = 0
            for a in A:
                A_mask |= (1 << a)
            remaining_mask = ((1 << N) - 1) ^ A_mask
            # Enumerate 4-subsets of remaining
            remaining = [v for v in range(N) if (remaining_mask >> v) & 1]
            for B in itertools.combinations(remaining, 4):
                # Check edge count between A and B is exactly 12
                # For each a in A, count neighbors in B
                total = 0
                for a in A:
                    mask_a = masks[a]
                    count = 0
                    for b in B:
                        if (mask_a >> b) & 1:
                            count += 1
                    total += count
                if total != 12:
                    continue
                # Check missing edges form perfect matching
                missing = {}
                ok = True
                for a in A:
                    mask_a = masks[a]
                    miss = None
                    for b in B:
                        if not ((mask_a >> b) & 1):
                            if miss is not None:
                                ok = False
                                break
                            miss = b
                    if not ok:
                        break
                    if miss is None:
                        ok = False
                        break
                    missing[a] = miss
                if not ok:
                    continue
                # Check perfect matching
                seen_b = set(missing.values())
                if len(seen_b) == 4:
                    return True
        return False
    
    # Greedy removal
    while time.time() < deadline - 30:
        # Find a Q3 and remove one of its edges
        found = False
        for A in itertools.combinations(range(N), 4):
            A_mask = 0
            for a in A:
                A_mask |= (1 << a)
            remaining = [v for v in range(N) if not ((A_mask >> v) & 1)]
            for B in itertools.combinations(remaining, 4):
                total = 0
                for a in A:
                    for b in B:
                        if (adj_mask[a] >> b) & 1:
                            total += 1
                if total != 12:
                    continue
                missing = {}
                ok = True
                for a in A:
                    miss = None
                    for b in B:
                        if not ((adj_mask[a] >> b) & 1):
                            if miss is not None:
                                ok = False
                                break
                            miss = b
                    if not ok:
                        break
                    if miss is None:
                        ok = False
                        break
                    missing[a] = miss
                if not ok:
                    continue
                if len(set(missing.values())) == 4:
                    # Remove edge (A[0], B[0])
                    u, v = A[0], B[0]
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                    found = True
                    break
            if found:
                break
        if not found:
            break
    
    # Count edges
    edge_count = sum(bin(adj_mask[u]).count('1') for u in range(N)) // 2
    
    # Try to add back edges
    for u in range(N):
        for v in range(u + 1, N):
            if time.time() > deadline - 5:
                break
            if (adj_mask[u] >> v) & 1:
                continue
            # Try adding
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            if not has_q3_fast(adj_mask):
                pass  # keep it
            else:
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
        if time.time() > deadline - 5:
            break
    
    # Build result
    result = []
    for u in range(N):
        for v in range(u + 1, N):
            if (adj_mask[u] >> v) & 1:
                result.append([u, v])
    
    # Fallback to seed if too few edges
    if len(result) < 39:
        result = [[a, b] for a in range(3) for b in range(3, 16)]
    
    return result
# EVOLVE-BLOCK-END
