"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def build_partition(n, k, p, seed):
        # Generate the graph
        rng = random.Random(seed)
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        
        # Build neighbor sets for independent vertices
        # N[v] is set of u in [0, k) such that (u, v) is an edge
        N = {}
        for v in range(k, n):
            N[v] = set()
        for u, v in cross_edges:
            N[v].add(u)
            
        # Total cross edges
        E_cross = len(cross_edges)
        
        # Strategy:
        # Partition C = [0, k) into m sets T_1...T_m.
        # Assign distinct v_i in I to T_i such that T_i subset N[v_i].
        # Cliques: {v_i} U T_i for each i.
        # Remaining cross edges: {u, w}.
        # Total: m + (E_cross - sum(|T_i|)) = m + E_cross - k.
        
        # We want to minimize m.
        # Try m=1, then m=2, etc.
        
        # To find if a set T is covered by some v, we check if T is subset of N[v].
        
        # Greedy approach to find partition:
        # Start with C.
        # If covered by some v, done (m=1).
        # Else, split C into two halves.
        # Check if each half is covered by some distinct v.
        # If yes, done (m=2).
        # Else, split further.
        
        # We need to be careful about time.
        
        # Let's just try to find a matching for a fixed partition size m.
        # But partitioning C is also a choice.
        
        # Simple heuristic:
        # 1. Try m=1.
        #    Find v such that C subset N[v].
        # 2. If fail, try m=2.
        #    Split C into T1, T2.
        #    Find v1, v2 distinct such that T1 subset N[v1] and T2 subset N[v2].
        # 3. If fail, try m=3... up to k.
        #    If m=k, T_i = {i}. Find v_i for each i.
        
        # How to split C?
        # Just split into contiguous blocks or random?
        # Contiguous is fine.
        
        # Let's implement a function to check if we can cover a list of sets with distinct v's.
        
        def can_cover(sets):
            # sets is a list of sets of vertices in C
            # returns list of v's or None
            # Use simple greedy matching
            # Sort sets by size descending?
            # For each set, find available v that covers it.
            
            available_v = set(range(k, n))
            assignment = []
            
            # Heuristic: process larger sets first
            sorted_sets = sorted(sets, key=len, reverse=True)
            
            for s in sorted_sets:
                found = False
                # Try to find a v in available_v such that s subset N[v]
                # To speed up, we can precompute which v's cover which subsets?
                # No, just iterate.
                for v in list(available_v):
                    if s.issubset(N[v]):
                        assignment.append(v)
                        available_v.remove(v)
                        found = True
                        break
                if not found:
                    return None
            return assignment
        
        # Try m from 1 to k
        best_cliques = None
        best_count = float('inf')
        
        for m in range(1, k + 1):
            if time.time() > deadline:
                break
                
            # Create m sets partitioning C
            # Simple split: C = [0, k)
            # T_i = { i*k//m ... (i+1)*k//m - 1 }
            sets = []
            for i in range(m):
                start = i * k // m
                end = (i + 1) * k // m
                sets.append(set(range(start, end)))
            
            # Check if we can cover these sets
            assignment = can_cover(sets)
            if assignment:
                # Build cliques
                cliques = []
                for i in range(m):
                    v = assignment[i]
                    cliques.append(list(sets[i]) + [v])
                
                # Count covered cross edges
                covered = 0
                for i in range(m):
                    covered += len(sets[i])
                
                # Add remaining cross edges
                # We need to identify which cross edges are covered
                # Covered edges are (u, v) where v is in assignment and u is in sets[assignment.index(v)]
                # Actually, simpler:
                # For each v in assignment, the covered edges are (u, v) for u in its assigned set.
                
                # Let's just add all cross edges not in the big cliques
                # To avoid checking each edge, we can just add all and remove?
                # No, we must output the list.
                
                # Let's create a set of covered edges
                covered_edges = set()
                for i in range(m):
                    v = assignment[i]
                    for u in sets[i]:
                        covered_edges.add((u, v))
                
                for u, v in cross_edges:
                    if (u, v) not in covered_edges:
                        cliques.append([u, v])
                        
                count = len(cliques)
                if count < best_count:
                    best_count = count
                    best_cliques = cliques
                    
                # If m=1 worked, it's likely optimal or near optimal.
                # We can break if m=1?
                # m=1 gives 1 + E - k.
                # m=2 gives 2 + E - k.
                # m=1 is always better if it works.
                if m == 1:
                    break

        if best_cliques is None:
            # Fallback: 1 big clique C + all cross edges
            cliques = []
            if k >= 2:
                cliques.append(list(range(k)))
            for u, v in cross_edges:
                cliques.append([u, v])
            best_cliques = cliques
            
        return best_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(build_partition(*spec))
    
    return results
# EVOLVE-BLOCK-END
