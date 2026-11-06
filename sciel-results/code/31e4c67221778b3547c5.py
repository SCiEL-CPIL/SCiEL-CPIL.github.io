<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    # Greedy: scan edges in order and keep one unless it completes a four-cycle.
    d, chosen, out = 7, set(), []
    for x in range(2 ** d):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                ok = True
                for j in range(d):
                    if j != i:
                        c, e = x ^ (1 << j), y ^ (1 << j)
                        if all(p in chosen for p in ((min(x, c), max(x, c)), (min(y, e), max(y, e)), (min(c, e), max(c, e)))):
                            ok = False
                            break
                if ok:
                    chosen.add((x, y))
                    out.append([x, y])
    return out
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    d = 7
    n_v = 1 << d
    total_edges = d * (1 << (d - 1))  # 448
    
    # Precompute all edges
    all_edges = []
    edge_to_idx = {}
    for x in range(n_v):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    n_e = len(all_edges)
    
    # Precompute for each edge, the list of other edges that form a 4-cycle with it
    # Edge (x,y) with dimension i: for each j != i, the cycle is (x, y, y^bit_j, x^bit_j)
    # The other three edges in the cycle are: (x, x^bit_j), (y, y^bit_j), (x^bit_j, y^bit_j)
    edge_cycles = [[] for _ in range(n_e)]
    
    for idx, (x, y) in enumerate(all_edges):
        # Find which dimension this edge is
        diff = x ^ y
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            bit_j = 1 << j
            c = x ^ bit_j
            e = y ^ bit_j
            # Three other edges in this 4-cycle
            e1 = (min(x, c), max(x, c))
            e2 = (min(y, e), max(y, e))
            e3 = (min(c, e), max(c, e))
            edge_cycles[idx].append((edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3]))
    
    best_count = 0
    best_solution = None
    
    def greedy_random(seed):
        rng = random.Random(seed)
        order = list(range(n_e))
        rng.shuffle(order)
        chosen = [False] * n_e
        count = 0
        for idx in order:
            x, y = all_edges[idx]
            ok = True
            for (a, b, c) in edge_cycles[idx]:
                if chosen[a] and chosen[b] and chosen[c]:
                    ok = False
                    break
            if ok:
                chosen[idx] = True
                count += 1
        return chosen, count
    
    def local_search(chosen, count, rng):
        # Try to improve by 1-swap moves
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each edge and see if we can add more
            for idx in range(n_e):
                if not chosen[idx]:
                    continue
                # Try removing this edge
                chosen[idx] = False
                count -= 1
                # Greedily try to add edges
                added = []
                for jdx in range(n_e):
                    if chosen[jdx]:
                        continue
                    ok = True
                    for (a, b, c) in edge_cycles[jdx]:
                        if chosen[a] and chosen[b] and chosen[c]:
                            ok = False
                            break
                    if ok:
                        chosen[jdx] = True
                        count += 1
                        added.append(jdx)
                # If we gained net, keep it
                if count > best_count or (not improved and count >= best_count):
                    improved = True
                    # Roll back if we didn't improve
                    if count < best_count:
                        for jdx in added:
                            chosen[jdx] = False
                        chosen[idx] = True
                        count = best_count
                    best_count = count
                else:
                    # Roll back
                    for jdx in added:
                        chosen[jdx] = False
                    chosen[idx] = True
                    count += 1
        return chosen, count
    
    # Multiple restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() >= deadline:
            break
        chosen, count = greedy_random(r * 1000 + 42)
        if count > best_count:
            best_count = count
            best_solution = chosen[:]
        
        # Local search
        if time.time() < deadline:
            rng = random.Random(r * 7 + 13)
            chosen, count = local_search(chosen, count, rng)
            if count > best_count:
                best_count = count
                best_solution = chosen[:]
    
    if best_solution is None:
        # Fallback: deterministic greedy
        chosen, count = greedy_random(0)
        best_solution = chosen
        best_count = count
    
    # Convert to edge list
    result = []
    for idx in range(n_e):
        if best_solution[idx]:
            x, y = all_edges[idx]
            result.append([x, y])
    
    return result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I realize the local search as written is O(n_e * n_e * cycles) which is way too slow. Let me redesign with a more efficient approach.

The key issue: for each removal, trying to add all edges is O(n_e * 5 * 3) per edge removal, and we have n_e edges to try removing. That's O(n_e^2 * 15) ≈ 448^2 * 15 ≈ 3 million operations per full pass, times many passes. Too slow.

Better approach: 
- For each edge not in the solution, check if it can be added
- If not, it's blocked by some 4-cycle
- Try a "flip": remove one edge from the blocking cycle, add the new edge
- This is a 1-for-1 swap that might enable further additions

Let me rewrite with a cleaner, faster local search:

