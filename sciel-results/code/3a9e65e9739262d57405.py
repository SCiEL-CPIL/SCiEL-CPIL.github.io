import math
import random

def construct():
    # Squares up to 256+256 = 512
    squares = {k*k for k in range(1, int(math.isqrt(512)) + 2)}
    
    # Precompute conflicts
    # conflicts[i] is a set of j such that i+j is a perfect square
    N = 256
    conflicts = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            if i + j in squares:
                conflicts[i].add(j)
                conflicts[j].add(i)
        
    # 1. Greedy Initial Solution
    # Sort by degree to pick min degree nodes first? 
    # Standard greedy for Max Independent Set: pick node with min degree in remaining graph.
    
    def get_greedy_set():
        remaining = set(range(1, N + 1))
        S = set()
        
        while remaining:
            # Find node with minimum degree in the induced subgraph of 'remaining'
            min_deg_node = None
            min_deg = float('inf')
            for u in remaining:
                deg = len(conflicts[u] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_deg_node = u
            
            S.add(min_deg_node)
            # Remove min_deg_node and its neighbors from remaining
            remaining.discard(min_deg_node)
            for v in conflicts[min_deg_node]:
                remaining.discard(v)
                
        return S

    S = get_greedy_set()
    
    # 2. Local Search: 1-for-2 swaps
    # If removing u allows adding >= 2 nodes, do it.
    
    def find_mis_in_candidates(candidates):
        # Find Maximal Independent Set in small set 'candidates'
        # candidates is a list of nodes
        # Use greedy on this small subgraph
        cand_set = set(candidates)
        local_S = set()
        while cand_set:
            # Pick node with min degree within cand_set
            min_deg_node = None
            min_deg = float('inf')
            for u in cand_set:
                deg = len(conflicts[u] & cand_set)
                if deg < min_deg:
                    min_deg = deg
                    min_deg_node = u
            
            local_S.add(min_deg_node)
            cand_set.discard(min_deg_node)
            for v in conflicts[min_deg_node]:
                cand_set.discard(v)
        return local_S

    improved = True
    while improved:
        improved = False
        # Iterate over a copy of S to allow modification
        for u in list(S):
            # Find nodes v not in S that conflict with S ONLY via u
            candidates = []
            for v in range(1, N + 1):
                if v in S:
                    continue
                # Check if v conflicts with any node in S \ {u}
                # If it does, v cannot be added even if u is removed.
                # If it only conflicts with u, v is a candidate.
                has_other_conflict = False
                for w in S:
                    if w == u:
                        continue
                    if v in conflicts[w]:
                        has_other_conflict = True
                        break
                if not has_other_conflict:
                    # Also ensure v actually conflicts with u (it must, otherwise it would have been in S? 
                    # Not necessarily, greedy might have skipped it. 
                    # But if v doesn't conflict with u, and doesn't conflict with S\{u}, 
                    # it doesn't conflict with S at all, so it should have been addable?
                    # Actually, if v doesn't conflict with S, it is an isolated node in the complement?
                    # Let's just check: if v is not in S and doesn't conflict with S, we can just add it.
                    # The greedy step produces a MAXIMAL independent set, so every v not in S 
                    # must conflict with at least one node in S.
                    # So v must conflict with u or S\{u}.
                    # If it doesn't conflict with S\{u}, it MUST conflict with u.
                    candidates.append(v)
            
            if len(candidates) < 2:
                continue
                
            # Find a maximal independent set in the subgraph induced by 'candidates'
            # Note: candidates are independent of S\{u}. 
            # We just need them to be independent among themselves.
            new_nodes = find_mis_in_candidates(candidates)
            
            if len(new_nodes) > 1:
                # Replace u with new_nodes
                S.remove(u)
                S.update(new_nodes)
                improved = True
                break # Restart the loop since S changed significantly
                
    return list(S)
