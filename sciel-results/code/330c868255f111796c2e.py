"""Construct a large (5,5)-free graph via local search on Cayley graphs."""
import random
from itertools import combinations

def construct():
    """
    Uses a randomized local search on Cayley graphs over Z_n to find large
    (5,5)-free graphs. Falls back to the Paley graph on 37 vertices if
    no larger Cayley graph is found.
    """
    CLIQUE = 5
    INDEPENDENT = 5
    MAX_VERTICES = 46

    def check_cayley(n, S):
        """
        Checks if the Cayley graph on Z_n with connection set S is (5,5)-free.
        Returns (has_clique5, has_indep5).
        """
        # Build adjacency list as list of sets for quick lookup
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = (j - i) % n
                    if diff in S:
                        adj[i].add(j)
                        adj[j].add(i)
        
        # Check for K5
        for v0 in range(n):
            nbs = list(adj[v0])
            if len(nbs) < 4:
                continue
            for combo in combinations(nbs, 4):
                # Check if combo is a K4
                is_k4 = True
                for a, b in combinations(combo, 2):
                    if b not in adj[a]:
                        is_k4 = False
                        break
                if is_k4:
                    return True, False # Found Clique
            
        # Check for Independent 5
        for v0 in range(n):
            non_nbs = [v for v in range(n) if v != v0 and v not in adj[v0]]
            if len(non_nbs) < 4:
                continue
            for combo in combinations(non_nbs, 4):
                # Check if combo is independent
                is_indep = True
                for a, b in combinations(combo, 2):
                    if b in adj[a]:
                        is_indep = False
                        break
                if is_indep:
                    return False, True # Found Independent Set
                    
        return False, False

    def paley_graph(n):
        """Constructs Paley graph on n vertices (n must be prime, n % 4 == 1)."""
        Q = set()
        for i in range(1, n):
            Q.add((i * i) % n)
        S = Q
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n in S:
                    edges.append((i, j))
        return n, edges

    best_n = 0
    best_edges = []

    # Try to find a valid Cayley graph for n from MAX_VERTICES down to 37
    for n in range(MAX_VERTICES, 36, -1):
        random.seed(n * 7919) # Deterministic seed per n
        
        # Initialize S: Random symmetric subset
        S = set()
        used = set()
        for x in range(1, n):
            if x in used:
                continue
            neg_x = (-x) % n
            if neg_x == x:
                if random.random() < 0.5:
                    S.add(x)
                used.add(x)
            else:
                if neg_x in used:
                    continue
                if random.random() < 0.5:
                    S.add(x)
                    S.add(neg_x)
                used.add(x)
                used.add(neg_x)
        
        current_S = set(S)
        
        # Candidates for flipping: pairs (x, -x)
        candidates = []
        for x in range(1, n):
            neg_x = (-x) % n
            if x < neg_x or (x == neg_x and x != 0):
                candidates.append((x, neg_x))
        
        random.shuffle(candidates)
        
        found = False
        # Run local search
        for _ in range(100):
            c1, c2 = check_cayley(n, current_S)
            if not c1 and not c2:
                found = True
                break
            
            if not candidates:
                break
                
            idx = random.randint(0, len(candidates) - 1)
            x, nx = candidates[idx]
            
            # Flip
            if x in current_S:
                current_S.remove(x)
            else:
                current_S.add(x)
            if nx in current_S:
                current_S.remove(nx)
            else:
                current_S.add(nx)
                
            new_c1, new_c2 = check_cayley(n, current_S)
            
            # Accept if not worse
            if (new_c1 + new_c2) <= (c1 + c2):
                pass
            else:
                # Revert
                if x in current_S:
                    current_S.remove(x)
                else:
                    current_S.add(x)
                if nx in current_S:
                    current_S.remove(nx)
                else:
                    current_S.add(nx)
                
        if found:
            best_n = n
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (j - i) % n in current_S or (i - j) % n in current_S:
                        edges.append((i, j))
            best_edges = edges
            break

    # Fallback to Paley 37 if no better found
    if best_n == 0:
        best_n, best_edges = paley_graph(37)

    return best_n, best_edges
