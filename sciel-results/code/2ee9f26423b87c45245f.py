"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, permutations

def construct():
    start = time.time()
    deadline = start + 140
    n = 16

    # Q3: 8 vertices, 12 edges
    q3_edges = []
    for u in range(8):
        for v in range(u + 1, 8):
            diff = u ^ v
            if diff & (diff - 1) == 0:
                q3_edges.append((u, v))
    assert len(q3_edges) == 12

    q3_adj = {v: set() for v in range(8)}
    for a, b in q3_edges:
        q3_adj[a].add(b)
        q3_adj[b].add(a)

    # All possible edges in K16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_edges)  # 120

    # Precompute Q3 adjacency info for fast checking
    # Q3 bipartition: A={0,3,5,6}, B={1,2,4,7}
    q3_A = [0, 3, 5, 6]
    q3_B = [1, 2, 4, 7]
    
    # For each a in A, neighbors in B
    q3_AB = {}
    for a in q3_A:
        q3_AB[a] = [b for b in q3_B if (a ^ b) & ((a ^ b) - 1) == 0]

    # Precompute all 24 permutations of 4 elements
    perms4 = list(permutations(range(4)))

    # Fast Q3 check using bitmasks
    def has_q3_fast(adj_bitmasks, u, v):
        """Check if adding edge (u,v) creates a Q3.
        adj_bitmasks[i] is bitmask of neighbors of i (NOT including the new edge (u,v) yet)."""
        # Map u->0, v->1 in Q3 (0^1=1, valid edge)
        # Also try u->1, v->0
        for mapping in [(u, v), (v, u)]:
            qv_to_kv = {0: mapping[0], 1: mapping[1]}
            used = {mapping[0], mapping[1]}
            
            mask0 = adj_bitmasks[mapping[0]]
            mask1 = adj_bitmasks[mapping[1]]
            
            # Remove used vertices
            mask0 &= ~(1 << mapping[0])
            mask0 &= ~(1 << mapping[1])
            mask1 &= ~(1 << mapping[0])
            mask1 &= ~(1 << mapping[1])
            
            # Get candidate lists
            cands2 = [i for i in range(n) if mask0 & (1 << i)]
            cands4 = [i for i in range(n) if mask0 & (1 << i)]
            cands3 = [i for i in range(n) if mask1 & (1 << i)]
            cands5 = [i for i in range(n) if mask1 & (1 << i)]
            
            # Try all combinations
            for c2 in cands2:
                if c2 in used:
                    continue
                for c4 in cands4:
                    if c4 == c2 or c4 in used:
                        continue
                    for c3 in cands3:
                        if c3 in used or c3 == c2 or c3 == c4:
                            continue
                        for c5 in cands5:
                            if c5 in used or c5 == c2 or c5 == c3 or c5 == c4:
                                continue
                                
                            # Now we have 6 vertices assigned: 0,1,2,3,4,5
                            # Need to assign 6 and 7
                            # Q3 vertex 6 connects to 2 and 4
                            # Q3 vertex 7 connects to 3 and 5
                            
                            # c6 must be neighbor of both c2 and c4
                            mask_c2 = adj_bitmasks[c2]
                            mask_c4 = adj_bitmasks[c4]
                            mask_c6 = mask_c2 & mask_c4
                            mask_c6 &= ~(1 << mapping[0])
                            mask_c6 &= ~(1 << mapping[1])
                            mask_c6 &= ~(1 << c2)
                            mask_c6 &= ~(1 << c3)
                            mask_c6 &= ~(1 << c4)
                            mask_c6 &= ~(1 << c5)
                            
                            cands6 = [i for i in range(n) if mask_c6 & (1 << i)]
                            
                            if not cands6:
                                continue
                            
                            # c7 must be neighbor of both c3 and c5
                            mask_c3 = adj_bitmasks[c3]
                            mask_c5 = adj_bitmasks[c5]
                            mask_c7 = mask_c3 & mask_c5
                            mask_c7 &= ~(1 << mapping[0])
                            mask_c7 &= ~(1 << mapping[1])
                            mask_c7 &= ~(1 << c2)
                            mask_c7 &= ~(1 << c3)
                            mask_c7 &= ~(1 << c4)
                            mask_c7 &= ~(1 << c5)
                            
                            cands7 = [i for i in range(n) if mask_c7 & (1 << i)]
                            
                            if not cands7:
                                continue
                            
                            for c6 in cands6:
                                for c7 in cands7:
                                    if c7 == c6:
                                        continue
                                    # Check all Q3 edges
                                    # (2,3): c2-c3
                                    if not (adj_bitmasks[c2] & (1 << c3)):
                                        continue
                                    # (2,6): c2-c6
                                    if not (adj_bitmasks[c2] & (1 << c6)):
                                        continue
                                    # (3,7): c3-c7
                                    if not (adj_bitmasks[c3] & (1 << c7)):
                                        continue
                                    # (4,5): c4-c5
                                    if not (adj_bitmasks[c4] & (1 << c5)):
                                        continue
                                    # (4,6): c4-c6
                                    if not (adj_bitmasks[c4] & (1 << c6)):
                                        continue
                                    # (5,7): c5-c7
                                    if not (adj_bitmasks[c5] & (1 << c7)):
                                        continue
                                    # (6,7): c6-c7
                                    if not (adj_bitmasks[c6] & (1 << c7)):
                                        continue
                                    
                                    return True
        return False

    # Build initial graph: K_{3,13}
    def build_initial():
        adj = [0] * n
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        edges = set()
        for a in range(3):
            for b in range(3, 16):
                edges.add((a, b))
        return adj, edges

    best_adj, best_edges = build_initial()
    best_count = len(best_edges)
    
    random.seed(42)
    
    # Greedy: try to add edges in random order
    for trial in range(30):
        if time.time() > deadline - 25:
            break
        
        adj, edges = build_initial()
        missing = [e for e in all_edges if e not in edges]
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline - 20:
                break
            u, v = e
            if has_q3_fast(adj, u, v):
                continue
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add(e)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
            best_adj = adj[:]

    # Genetic algorithm with repair
    if time.time() < deadline - 20:
        # Population of solutions
        population = [(best_adj[:], set(best_edges), best_count)]
        
        # Generate initial population
        for _ in range(5):
            if time.time() > deadline - 20:
                break
            adj, edges = build_initial()
            missing = [e for e in all_edges if e not in edges]
            random.shuffle(missing)
            for e in missing:
                if time.time() > deadline - 18:
                    break
                u, v = e
                if has_q3_fast(adj, u, v):
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add(e)
            population.append((adj, edges, len(edges)))
        
        # Sort by count descending
        population.sort(key=lambda x: -x[2])
        
        # Genetic evolution
        gen = 0
        while time.time() < deadline - 15 and gen < 50:
            gen += 1
            
            # Select parents (tournament selection)
            def tournament_select(pop, k=3):
                candidates = random.sample(pop, min(k, len(pop)))
                return max(candidates, key=lambda x: x[2])
            
            # Crossover: combine edge sets from two parents
            p1_adj, p1_edges, p1_count = tournament_select(population)
            p2_adj, p2_edges, p2_count = tournament_select(population)
            
            # Create child by taking union of edges, then repair
            child_edges = set(p1_edges) | set(p2_edges)
            
            # Repair: remove edges that create Q3
            # Build adjacency from child_edges
            child_adj = [0] * n
            for u, v in child_edges:
                child_adj[u] |= (1 << v)
                child_adj[v] |= (1 << u)
            
            # Check if child is valid; if not, remove conflicting edges
            # We need to find which edges to remove. Use a greedy approach:
            # Try removing each edge one at a time and see if it fixes things.
            
            # Actually, let's use a different repair strategy:
            # Start with empty graph and add edges from the union in random order,
            # skipping any that create Q3.
            
            child_adj = [0] * n
            child_edges = set()
            
            # Get all edges from both parents, ordered by some heuristic
            all_candidate_edges = list(p1_edges | p2_edges)
            random.shuffle(all_candidate_edges)
            
            for e in all_candidate_edges:
                if time.time() > deadline - 15:
                    break
                u, v = e
                if has_q3_fast(child_adj, u, v):
                    continue
                child_adj[u] |= (1 << v)
                child_adj[v] |= (1 << u)
                child_edges.add(e)
            
            child_count = len(child_edges)
            
            # Mutation: try to add random missing edges
            if time.time() < deadline - 15:
                missing = [e for e in all_edges if e not in child_edges]
                random.shuffle(missing)
                for e in missing[:10]:  # Try up to 10 random additions
                    if time.time() > deadline - 15:
                        break
                    u, v = e
                    if has_q3_fast(child_adj, u, v):
                        continue
                    child_adj[u] |= (1 << v)
                    child_adj[v] |= (1 << u)
                    child_edges.add(e)
                child_count = len(child_edges)
            
            # Update population
            population.append((child_adj, child_edges, child_count))
            
            # Keep only top 10
            population.sort(key=lambda x: -x[2])
            population = population[:10]
            
            # Update best
            if child_count > best_count:
                best_count = child_count
                best_edges = set(child_edges)
                best_adj = child_adj[:]

    # Local search: try removing one edge and adding another
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 8:
            improved = False
            adj = best_adj[:]
            edges = set(best_edges)
            
            # Try each edge removal
            for e in list(edges):
                if time.time() > deadline - 8:
                    break
                u, v = e
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges.remove(e)
                
                # Try to add a different edge
                missing = [ne for ne in all_edges if ne not in edges]
                random.shuffle(missing)
                
                for ne in missing:
                    if time.time() > deadline - 8:
                        break
                    nu, nv = ne
                    if has_q3_fast(adj, nu, nv):
                        continue
                    adj[nu] |= (1 << nv)
                    adj[nv] |= (1 << nu)
                    edges.add(ne)
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = set(edges)
                        best_adj = adj[:]
                        improved = True
                    break  # Only try one addition per removal

    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
