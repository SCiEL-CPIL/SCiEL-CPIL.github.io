import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    def get_paley(q):
        """Generate Paley graph on q vertices where q is a prime ≡ 1 (mod 4)."""
        residues = {(x * x) % q for x in range(1, q)}
        n = q
        adj = [set() for _ in range(n)]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % q in residues:
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.append((i, j))
        return n, adj, edges
    
    def is_5_5_free(n, adj):
        """Check if graph has no K5 and no independent set of size 5."""
        # Check for K5
        for comb in itertools.combinations(range(n), 5):
            is_clique = True
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = comb[i], comb[j]
                    if v not in adj[u]:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                return False
        # Check for independent set of size 5
        for comb in itertools.combinations(range(n), 5):
            is_independent = True
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = comb[i], comb[j]
                    if v in adj[u]:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                return False
        return True
    
    def find_k5s(n, adj):
        """Find all K5 subgraphs."""
        k5s = []
        for comb in itertools.combinations(range(n), 5):
            is_clique = True
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = comb[i], comb[j]
                    if v not in adj[u]:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                k5s.append(comb)
        return k5s
    
    def find_independent_5s(n, adj):
        """Find all independent sets of size 5."""
        ind5s = []
        for comb in itertools.combinations(range(n), 5):
            is_independent = True
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = comb[i], comb[j]
                    if v in adj[u]:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                ind5s.append(comb)
        return ind5s
    
    def check_and_get_edges(n, adj):
        """Check if graph is (5,5)-free and return edges if so."""
        if is_5_5_free(n, adj):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if j in adj[i]:
                        edges.append((i, j))
            return True, edges
        return False, None
    
    # First, try Paley 37 as baseline
    best_n = 37
    best_edges = None
    
    n37, adj37, _ = get_paley(37)
    ok, edges37 = check_and_get_edges(n37, adj37)
    if ok:
        best_n = n37
        best_edges = edges37
    
    # Try to improve by removing a vertex from Paley 43
    if time.time() < deadline:
        n43, adj43, _ = get_paley(43)
        
        # Find all K5 and independent 5-sets in Paley 43
        k5s = find_k5s(n43, adj43)
        ind5s = find_independent_5s(n43, adj43)
        
        # Union of vertices in any obstruction
        obstruction_vertices = set()
        for comb in k5s:
            obstruction_vertices.update(comb)
        for comb in ind5s:
            obstruction_vertices.update(comb)
        
        # Try removing each vertex in obstruction_vertices
        for v in sorted(obstruction_vertices):
            if time.time() > deadline:
                break
            
            # Build 42-vertex graph by removing vertex v
            new_n = 42
            new_adj = [set() for _ in range(new_n)]
            
            # Map old indices to new indices
            old_to_new = {}
            new_idx = 0
            for i in range(n43):
                if i != v:
                    old_to_new[i] = new_idx
                    new_idx += 1
            
            # Copy edges
            for i in range(n43):
                if i == v:
                    continue
                for j in adj43[i]:
                    if j != v:
                        ni, nj = old_to_new[i], old_to_new[j]
                        if ni < nj:
                            new_adj[ni].add(nj)
                            new_adj[nj].add(ni)
            
            # Check if this 42-vertex graph is (5,5)-free
            ok, edges = check_and_get_edges(new_n, new_adj)
            if ok:
                best_n = new_n
                best_edges = edges
                break  # Found a valid 42-vertex graph
    
    # If we didn't find a 42-vertex graph, try other approaches
    if best_n < 42 and time.time() < deadline:
        # Try Paley graphs on other primes
        for q in [41, 53, 59, 61, 73, 89, 97]:
            if time.time() > deadline:
                break
            n, adj, _ = get_paley(q)
            ok, edges = check_and_get_edges(n, adj)
            if ok and n > best_n:
                best_n = n
                best_edges = edges
    
    return best_n, best_edges
