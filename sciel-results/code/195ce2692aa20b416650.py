import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        cliques = []
        remaining_edges = set(edges)
        
        # Pre-compute adjacency for initial state? No, graph changes.
        # We will maintain a set of remaining edges.
        
        while remaining_edges:
            if time.time() > deadline:
                # Emergency fallback: return individual edges for remaining
                # This might be invalid if we don't handle it carefully, 
                # but better than timeout. 
                # Actually, we should just return what we have if we can't finish?
                # No, the result must be valid.
                # If we break early, we must ensure all edges are covered.
                # So if we hit deadline, we should just add all remaining edges as 2-cliques.
                for e in remaining_edges:
                    cliques.append(list(e))
                break

            # Find a vertex with max degree in remaining_edges
            degrees = {v: 0 for v in range(n)}
            for u, v in remaining_edges:
                degrees[u] += 1
                degrees[v] += 1
            
            # If no edges, loop ends
            if not any(degrees.values()):
                break

            # Pick start vertex
            start_v = max(degrees, key=degrees.get)
            
            # Build clique greedily
            clique = [start_v]
            # Candidates: neighbors of start_v in remaining_edges
            candidates = set()
            for u, v in remaining_edges:
                if u == start_v:
                    candidates.add(v)
                elif v == start_v:
                    candidates.add(u)
            
            # We want to add vertices to clique that are connected to all current members
            # and preferably have high degree
            while candidates:
                # Pick candidate with max degree in remaining_edges
                # (Approximation: just max degree in remaining_edges overall)
                # Or better: max degree in the subgraph induced by candidates + clique?
                # Let's stick to max degree in remaining_edges for simplicity and speed.
                best_u = max(candidates, key=lambda x: degrees.get(x, 0))
                
                # Check if best_u is connected to all in clique
                is_connected = True
                for c in clique:
                    if (min(best_u, c), max(best_u, c)) not in remaining_edges:
                        is_connected = False
                        break
                
                if is_connected:
                    clique.append(best_u)
                    candidates.remove(best_u)
                    # Update candidates: only keep those connected to best_u
                    # This is O(|candidates| * |clique|) roughly
                    new_candidates = set()
                    for u in candidates:
                        # Check if u is connected to best_u in remaining_edges
                        if (min(u, best_u), max(u, best_u)) in remaining_edges:
                            # Also must be connected to all other clique members?
                            # Since we only add if connected to ALL, and we filter at each step,
                            # if u was in candidates, it was connected to all previous clique members.
                            # Now we just need it to be connected to best_u.
                            new_candidates.add(u)
                    candidates = new_candidates
                else:
                    # If not connected, remove it and try next best?
                    # Or just remove it and continue?
                    # If we remove it, we might miss a large clique.
                    # But checking all candidates for connectivity is expensive.
                    # Greedy: just remove the one that failed?
                    # Actually, if best_u failed, it's not connected to some c in clique.
                    # So it can't be in the clique. Remove it.
                    candidates.remove(best_u)
            
            if len(clique) < 2:
                # Fallback: just take any edge
                e = next(iter(remaining_edges))
                clique = [e[0], e[1]]
            
            cliques.append(clique)
            
            # Remove all edges in clique from remaining_edges
            for u, v in combinations(clique, 2):
                edge = (min(u, v), max(u, v))
                remaining_edges.discard(edge)
                
        return cliques

    # Instance 1
    cliques1 = solve_instance(32, 0.6, 601)
    
    # Check time
    if time.time() > deadline:
         # If we are already over, we can't do instance 2 properly.
         # But we must return two lists.
         # Let's just return empty or minimal for the second one? No, must be valid.
         # The problem says "two lists... one per instance".
         # If we time out, we should probably just return the best we can.
         # But if we haven't solved instance 2, we can't return a valid partition for it.
         # However, the loop inside solve_instance has a deadline check that dumps remaining edges.
         # So solve_instance will return a valid partition (potentially suboptimal) even if it hits the deadline.
         # So we can just call it.
         pass

    # Instance 2
    cliques2 = solve_instance(64, 0.55, 602)

    return [cliques1, cliques2]
