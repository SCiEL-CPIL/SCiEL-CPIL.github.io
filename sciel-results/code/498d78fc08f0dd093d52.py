"""
K2,3-free graph construction on 20 vertices maximizing edges.

Strategy:
- Use the complement graph formulation: K2,3-free <=> complement has no P4
  (no 3-vertex path in complement).
- Use bit-parallel representation for fast validity checks.
- Multi-restart local search with:
    * Greedy edge addition
    * Simulated annealing with swap moves
    * Exhaustive final polish
- Return the best graph found within the time budget.
"""

import time
import random
import math

def construct():
    n = 20
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    # Bitset helpers
    def popcount(x):
        return bin(x).count('1')

    def count_edges(adj):
        return sum(popcount(x) for x in adj) // 2

    def is_k23_free(adj):
        """Check if graph is K2,3-free using complement P4-free check."""
        # Complement adjacency
        comp = [0] * n
        for u in range(n):
            for v in range(n):
                if u != v and not (adj[u] & (1 << v)):
                    comp[u] |= (1 << v)
        # Check no P4 in complement: no 3-vertex path
        # A P4 is a-b-c-d where all consecutive pairs are edges in comp
        # Equivalent: no 4 vertices a,b,c,d with a-b, b-c, c-d in comp
        # and a-c, a-d, b-d NOT all in comp (to avoid K4, but K4 contains P4)
        # Actually, we just need to check: for every edge (u,v) in comp,
        # for every neighbor w of v in comp (w != u), for every neighbor x of w in comp (x != v):
        #   if x != u, then we have a P4: u-v-w-x
        # Wait, that's not quite right. A P4 is a path of length 3.
        # Let me use a direct check: for all pairs (u,v), count common neighbors in comp.
        # If any pair has >= 2 common neighbors in comp, that doesn't directly give P4.
        # 
        # Actually, the standard characterization: G is K2,3-free iff 
        # for every pair of vertices {u,v}, |N(u) ∩ N(v)| <= 2.
        # This is the direct definition. Let me just use that.
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj[u] & adj[v]) >= 3:
                    return False
        return True

    def valid_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        if adj[u] & (1 << v):
            return False
        # 1. u and v must have at most 2 common neighbors
        if popcount(adj[u] & adj[v]) > 2:
            return False
        # 2. For each w in N(u), w and v must have at most 1 common neighbor
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if popcount(adj[w] & adj[v]) > 1:
                return False
        # 3. For each w in N(v), w and u must have at most 1 common neighbor
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if popcount(adj[w] & adj[u]) > 1:
                return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_edges(adj):
        edges = []
        for u in range(n):
            temp = adj[u]
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append((u, v))
                temp -= lsb
        return edges

    def get_non_edges(adj):
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    non_edges.append((u, v))
        return non_edges

    def greedy_fill(adj):
        """Greedily add all possible edges in random order."""
        non_edges = get_non_edges(adj)
        random.shuffle(non_edges)
        for u, v in non_edges:
            if valid_add(adj, u, v):
                add_edge(adj, u, v)

    # Initial seed: 5 disjoint K4s
    def make_k4_union():
        adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    add_edge(adj, base + a, base + b)
        return adj

    best_edges = 0
    best_adj = None

    # First, establish baseline
    adj0 = make_k4_union()
    greedy_fill(adj0)
    best_edges = count_edges(adj0)
    best_adj = adj0[:]

    # Multi-restart simulated annealing
    num_restarts = 30

    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break

        # Different starting points
        if r == 0:
            adj = make_k4_union()
        elif r % 3 == 1:
            # Start from best found, then perturb
            adj = best_adj[:]
            edges = get_edges(adj)
            random.shuffle(edges)
            for u, v in edges[:4]:
                remove_edge(adj, u, v)
        else:
            # Start from random K4 union with random perturbations
            adj = make_k4_union()
            # Add some random cross-edges if valid
            non_edges = get_non_edges(adj)
            random.shuffle(non_edges)
            for u, v in non_edges[:5]:
                if valid_add(adj, u, v):
                    add_edge(adj, u, v)

        greedy_fill(adj)
        current_edges = count_edges(adj)

        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

        # Simulated annealing
        T = 0.8
        T_min = 0.02
        cooling = 0.993
        steps = 0

        while T > T_min and (time.time() - start_time < time_limit):
            steps += 1
            move_type = random.random()

            if move_type < 0.40:
                # Try to add a random non-edge
                non_edges = get_non_edges(adj)
                if not non_edges:
                    T *= cooling
                    continue
                u, v = random.choice(non_edges)
                if valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]

            elif move_type < 0.55:
                # 1-for-2 swap: remove one edge, try to add two non-edges (net +1)
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or len(non_edges) < 2:
                    T *= cooling
                    continue
                u1, v1 = random.choice(edges)
                # Pick two distinct non-edges
                idx1 = random.randrange(len(non_edges))
                idx2 = random.randrange(len(non_edges) - 1)
                if idx2 >= idx1:
                    idx2 += 1
                u2, v2 = non_edges[idx1]
                u3, v3 = non_edges[idx2]
                
                # Remove (u1,v1)
                remove_edge(adj, u1, v1)
                
                # Try to add both (u2,v2) and (u3,v3)
                added_count = 0
                if valid_add(adj, u2, v2):
                    add_edge(adj, u2, v2)
                    added_count += 1
                if added_count == 1 and valid_add(adj, u3, v3):
                    add_edge(adj, u3, v3)
                    added_count += 1
                
                if added_count >= 1:
                    current_edges += added_count - 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    # Revert: re-add (u1,v1)
                    add_edge(adj, u1, v1)

            elif move_type < 0.65:
                # 1-for-3 swap: remove one edge, try to add three non-edges (net +2)
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or len(non_edges) < 3:
                    T *= cooling
                    continue
                u1, v1 = random.choice(edges)
                idxs = random.sample(range(len(non_edges)), 3)
                u2, v2 = non_edges[idxs[0]]
                u3, v3 = non_edges[idxs[1]]
                u4, v4 = non_edges[idxs[2]]
                
                remove_edge(adj, u1, v1)
                
                added_count = 0
                for eu, ev in [(u2, v2), (u3, v3), (u4, v4)]:
                    if valid_add(adj, eu, ev):
                        add_edge(adj, eu, ev)
                        added_count += 1
                
                if added_count >= 1:
                    current_edges += added_count - 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    add_edge(adj, u1, v1)

            elif move_type < 0.80:
                # 2-for-2 swap: remove two edges, try to add two non-edges (net 0)
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if len(edges) < 2 or len(non_edges) < 2:
                    T *= cooling
                    continue
                idx1 = random.randrange(len(edges))
                idx2 = random.randrange(len(edges) - 1)
                if idx2 >= idx1:
                    idx2 += 1
                u1, v1 = edges[idx1]
                u2, v2 = edges[idx2]
                idx3 = random.randrange(len(non_edges))
                idx4 = random.randrange(len(non_edges) - 1)
                if idx4 >= idx3:
                    idx4 += 1
                u3, v3 = non_edges[idx3]
                u4, v4 = non_edges[idx4]
                
                remove_edge(adj, u1, v1)
                remove_edge(adj, u2, v2)
                
                added_count = 0
                if valid_add(adj, u3, v3):
                    add_edge(adj, u3, v3)
                    added_count += 1
                if added_count == 1 and valid_add(adj, u4, v4):
                    add_edge(adj, u4, v4)
                    added_count += 1
                
                if added_count == 2:
                    # Net 0, accept with probability
                    if random.random() < 0.7:
                        pass
                    else:
                        remove_edge(adj, u3, v3)
                        remove_edge(adj, u4, v4)
                        add_edge(adj, u1, v1)
                        add_edge(adj, u2, v2)
                elif added_count == 1:
                    # Net -1, accept with SA probability
                    if random.random() < math.exp(-1.0 / T):
                        pass
                    else:
                        remove_edge(adj, u3, v3)
                        add_edge(adj, u1, v1)
                        add_edge(adj, u2, v2)
                else:
                    add_edge(adj, u1, v1)
                    add_edge(adj, u2, v2)

            elif move_type < 0.90:
                # Try to remove a random edge (accept with SA probability)
                edges = get_edges(adj)
                if not edges:
                    T *= cooling
                    continue
                u, v = random.choice(edges)
                if random.random() < math.exp(-1.0 / T):
                    remove_edge(adj, u, v)
                    current_edges -= 1

            else:
                # 1-for-1 swap: remove one edge, try to add another (net 0)
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or not non_edges:
                    T *= cooling
                    continue
                u1, v1 = random.choice(edges)
                u2, v2 = random.choice(non_edges)
                remove_edge(adj, u1, v1)

                if valid_add(adj, u2, v2):
                    add_edge(adj, u2, v2)
                    if random.random() < 0.7:
                        pass
                    else:
                        remove_edge(adj, u2, v2)
                        add_edge(adj, u1, v1)
                else:
                    add_edge(adj, u1, v1)

            T *= cooling

    # Final greedy fill on best
    if best_adj:
        greedy_fill(best_adj)
        final_edges = count_edges(best_adj)
        if final_edges > best_edges:
            best_edges = final_edges

    # Polish pass: try all possible single-edge additions and multi-edge swaps
    if best_adj and (time.time() - start_time < time_limit - 3):
        polish_adj = best_adj[:]
        polish_edges = count_edges(polish_adj)
        improved = True
        polish_iter = 0
        while improved and (time.time() - start_time < time_limit - 3):
            polish_iter += 1
            improved = False
            # Try adding any non-edge (multiple passes)
            for _ in range(5):
                any_added = False
                for u in range(n):
                    for v in range(u + 1, n):
                        if not (polish_adj[u] & (1 << v)):
                            if valid_add(polish_adj, u, v):
                                add_edge(polish_adj, u, v)
                                polish_edges += 1
                                improved = True
                                any_added = True
                if not any_added:
                    break
            
            # Try 1-for-2 swaps
            if not improved:
                edges = get_edges(polish_adj)
                non_edges = get_non_edges(polish_adj)
                if len(non_edges) >= 2:
                    for a, b in edges:
                        if time.time() - start_time > time_limit - 3:
                            break
                        for i in range(len(non_edges)):
                            for j in range(i + 1, len(non_edges)):
                                c, d = non_edges[i]
                                e, f = non_edges[j]
                                if (a, b) in [(c, d), (d, c), (e, f), (f, e)]:
                                    continue
                                remove_edge(polish_adj, a, b)
                                added = 0
                                if valid_add(polish_adj, c, d):
                                    add_edge(polish_adj, c, d)
                                    added += 1
                                if added == 1 and valid_add(polish_adj, e, f):
                                    add_edge(polish_adj, e, f)
                                    added += 1
                                if added >= 1:
                                    polish_edges += added - 1
                                    improved = True
                                    # Greedy fill
                                    for x in range(n):
                                        for y in range(x + 1, n):
                                            if not (polish_adj[x] & (1 << y)):
                                                if valid_add(polish_adj, x, y):
                                                    add_edge(polish_adj, x, y)
                                                    polish_edges += 1
                                                    improved = True
                                    break
                                else:
                                    add_edge(polish_adj, a, b)
                            if improved:
                                break
                        if improved:
                            break
            
            # Try 1-for-3 swaps
            if not improved and len(non_edges) >= 3:
                edges = get_edges(polish_adj)
                non_edges = get_non_edges(polish_adj)
                for a, b in edges:
                    if time.time() - start_time > time_limit - 3:
                        break
                    idxs = random.sample(range(len(non_edges)), min(3, len(non_edges)))
                    for i in range(len(idxs)):
                        for j in range(i+1, len(idxs)):
                            for k in range(j+1, len(idxs)):
                                c, d = non_edges[idxs[i]]
                                e, f = non_edges[idxs[j]]
                                g, h = non_edges[idxs[k]]
                                if (a, b) in [(c,d),(d,c),(e,f),(f,e),(g,h),(h,g)]:
                                    continue
                                remove_edge(polish_adj, a, b)
                                added = 0
                                for eu, ev in [(c,d),(e,f),(g,h)]:
                                    if valid_add(polish_adj, eu, ev):
                                        add_edge(polish_adj, eu, ev)
                                        added += 1
                                if added >= 1:
                                    polish_edges += added - 1
                                    improved = True
                                    for x in range(n):
                                        for y in range(x + 1, n):
                                            if not (polish_adj[x] & (1 << y)):
                                                if valid_add(polish_adj, x, y):
                                                    add_edge(polish_adj, x, y)
                                                    polish_edges += 1
                                                    improved = True
                                    break
                                else:
                                    add_edge(polish_adj, a, b)
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if polish_edges > best_edges:
                best_edges = polish_edges
                best_adj = polish_adj[:]

    # Final verification
    if best_adj:
        if not is_k23_free(best_adj):
            adj0 = make_k4_union()
            greedy_fill(adj0)
            best_adj = adj0

    # Convert to list of edges
    result = []
    for u, v in get_edges(best_adj):
        result.append([u, v])
    return result
