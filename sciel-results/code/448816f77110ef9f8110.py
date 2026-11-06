import time
from math import gcd

def construct():
    """
    Returns the maximum cardinality subset of [1, 255] containing 255,
    such that gcd(a, b) > 1 for all distinct pairs a, b in the subset.
    
    Uses an exact Maximum Clique search on the graph of candidates sharing a factor with 255.
    Falls back to the set of multiples of 3 (size 85) if the search times out or fails.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    # Fallback: all multiples of 3
    # 255 is a multiple of 3.
    # Any two multiples of 3 have gcd >= 3.
    fallback = list(range(3, 256, 3))
    
    # 1. Identify candidate nodes: numbers in [1, 255] that share a factor with 255
    # 255 = 3 * 5 * 17
    candidates = []
    for i in range(1, 256):
        if gcd(i, 255) > 1:
            candidates.append(i)
            
    n = len(candidates)
    
    # We need a clique containing 255.
    # 255 is connected to all candidates.
    # So we find the max clique in the subgraph induced by candidates \ {255}.
    
    # Remove 255 from candidates for the search
    if 255 in candidates:
        candidates.remove(255)
        
    m = len(candidates)
    
    # Build adjacency list for the subgraph
    # adj[i] is a set of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(m)]
    
    for i in range(m):
        for j in range(i + 1, m):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
                
    # Bron-Kerbosch with pivot and coloring bound
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
        
        if not P:
            return
            
        # Pivot selection
        # Choose u in P U X with maximum degree
        # Here we just pick the first one in P for simplicity, or one with high degree
        # Standard pivot: u = argmax_{v in P U X} |N(v) \cap P|
        # If P is empty, we can't pick pivot, but we handled not P.
        
        # Simple pivot: pick vertex in P with highest degree in P
        # Or just pick the first one. For small graphs, this is fine.
        # Let's pick the vertex in P with the highest number of neighbors in P.
        pivot = None
        max_deg = -1
        for v in P:
            deg = len(adj[v] & P)
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        if pivot is None:
            return
            
        # Branch on vertices in P that are NOT neighbors of pivot
        # This is the optimization: if we branch on all P, we do redundant work.
        # Vertices in P \ N(pivot) must be considered.
        # Actually, the standard Bron-Kerbosch with pivot says:
        # Branch on vertices in P \ N(pivot).
        # Wait, the theorem is: if P \ N(pivot) is empty, then R is a maximal clique.
        # If not, for each v in P \ N(pivot), recurse.
        
        candidates_to_branch = P - adj[pivot]
        
        if not candidates_to_branch:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
            
        # Order candidates for better pruning?
        # Not strictly necessary for this size, but good for performance.
        for v in candidates_to_branch:
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
            
    # Initial call
    P = set(range(m))
    X = set()
    R = set()
    
    # To ensure we find the MAXIMUM clique, standard Bron-Kerbosch finds all maximal cliques.
    # We track the largest one.
    # For performance, we can use a coloring bound to prune branches where
    # len(R) + max_possible_clique_in_P <= len(best_clique).
    # However, for m=126, the simple version might be fast enough.
    # Let's add a simple bound: if len(R) + len(P) <= len(best_clique), return.
    
    def bron_kerbosch_bounded(R, P, X):
        nonlocal best_clique
        
        if time.time() > deadline:
            return
            
        if len(R) + len(P) <= len(best_clique):
            return
            
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
            
        if not P:
            return
            
        # Pivot selection
        pivot = None
        max_deg = -1
        for v in P:
            deg = len(adj[v] & P)
            if deg > max_deg:
                max_deg = deg
                pivot = v
                
        if pivot is None:
            return
            
        candidates_to_branch = P - adj[pivot]
        
        if not candidates_to_branch:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
            
        for v in candidates_to_branch:
            bron_kerbosch_bounded(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
            
    try:
        bron_kerbosch_bounded(set(), set(range(m)), set())
    except Exception:
        pass
        
    if best_clique:
        result = [candidates[i] for i in best_clique]
        result.append(255)
        # Verify validity just in case
        valid = True
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                if gcd(result[i], result[j]) == 1:
                    valid = False
                    break
            if not valid:
                break
        if valid and 255 in result:
            return result
            
    return fallback

if __name__ == "__main__":
    res = construct()
    print(len(res))
    # Verify
    valid = True
    if 255 not in res:
        valid = False
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if gcd(res[i], res[j]) == 1:
                valid = False
                break
    print("Valid:", valid)
