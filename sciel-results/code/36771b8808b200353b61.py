import time
import random
import sys

def construct():
    deadline = time.time() + 115  # Safety margin under 150s kill
    
    def to_edges(n, A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (A[j] >> i) & 1:
                    edges.append([i, j])
        return edges
    
    def has_k4(n, A):
        """Check if graph contains a K4. Returns True if found."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                for c in range(b + 1, n):
                    if not ((A[c] >> a) & 1 and (A[c] >> b) & 1):
                        continue
                    for d in range(c + 1, n):
                        if (A[d] >> a) & 1 and (A[d] >> b) & 1 and (A[d] >> c) & 1:
                            return True
        return False
    
    def find_k4(n, A):
        """Find a K4 in the graph. Returns tuple (a,b,c,d) or None."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                for c in range(b + 1, n):
                    if not ((A[c] >> a) & 1 and (A[c] >> b) & 1):
                        continue
                    for d in range(c + 1, n):
                        if (A[d] >> a) & 1 and (A[d] >> b) & 1 and (A[d] >> c) & 1:
                            return (a, b, c, d)
        return None
    
    def max_independent_set_size(n, A, limit=6):
        """Find maximum independent set size using Bron-Kerbosch on complement.
        Returns size (capped at limit) and the set if size >= limit."""
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
        
        best_size = [0]
        best_set = [None]
        
        def bk(R, P, X):
            if len(R) > best_size[0]:
                best_size[0] = len(R)
                if len(R) >= limit:
                    best_set[0] = tuple(R)
                    return
            if not P:
                return
            if len(R) + bin(P).count("1") <= best_size[0]:
                return
            
            # Pivot selection
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                max_deg = -1
                u = -1
                p_temp = P
                while p_temp:
                    v_bit = p_temp & -p_temp
                    v = v_bit.bit_length() - 1
                    deg = bin(comp[v] & P).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                    p_temp ^= v_bit
                if u < 0:
                    candidates = P
                else:
                    candidates = P
            
            c = candidates
            while c:
                low = c & -c
                v = low.bit_length() - 1
                R.append(v)
                bk(R, P & comp[v], X & comp[v])
                R.pop()
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], all_mask, 0)
        return best_size[0], best_set[0]
    
    def has_ind6(n, A):
        """Check if graph has an independent set of size 6."""
        size, _ = max_independent_set_size(n, A, limit=6)
        return size >= 6
    
    def valid(n, A):
        if has_k4(n, A):
            return False
        if has_ind6(n, A):
            return False
        return True
    
    def make_paley(q):
        """Create Paley graph of order q (q must be prime ≡ 1 mod 4)."""
        if q < 5:
            return None
        for i in range(2, int(q**0.5) + 1):
            if q % i == 0:
                return None
        if q % 4 != 1:
            return None
        squares = {x * x % q for x in range(1, q)}
        n = q
        A = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and (j - i) % q in squares:
                    A[i] |= (1 << j)
        return A
    
    def make_cyclic_symmetric(n, D_half):
        """Create circulant graph on Z_n with symmetric difference set.
        D_half contains elements in 1..n//2, and we add both d and n-d."""
        D = set()
        for d in D_half:
            d = d % n
            if d != 0:
                D.add(d)
                D.add((n - d) % n)
        A = [0] * n
        for i in range(n):
            for d in D:
                j = (i + d) % n
                if j != i:
                    A[i] |= (1 << j)
        return A
    
    def schlaefli_graph(n, D):
        """Create Schläfli-type graph: vertex i connected to j if (j-i) mod n in D or (i-j) mod n in D.
        D should be a symmetric difference set."""
        A = [0] * n
        for i in range(n):
            for d in D:
                j = (i + d) % n
                if j != i:
                    A[i] |= (1 << j)
        return A
    
    def random_adj(n, p, rng):
        """Generate random graph with edge probability p."""
        A = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    A[i] |= (1 << j)
                    A[j] |= (1 << i)
        return A
    
    def try_local_search(n, A, rng, deadline, max_iters=500):
        """Local search: try to fix violations by flipping edges."""
        A = A[:]
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Check for K4
            k4 = find_k4(n, A)
            if k4:
                a, b, c, d = k4
                # Remove a random edge from the K4
                edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                u, v = rng.choice(edges)
                A[u] &= ~(1 << v)
                A[v] &= ~(1 << u)
                continue
            
            # Check for independent set of size 6
            size, ind_set = max_independent_set_size(n, A, limit=6)
            if size >= 6:
                # Add a random edge within the independent set
                verts = list(ind_set)
                u, v = rng.sample(verts, 2)
                A[u] |= (1 << v)
                A[v] |= (1 << u)
                continue
            
            # No violations - valid!
            return A
        
        return A
    
    def greedy_grow(start_A, start_n, target_n, rng, deadline):
        """Try to grow a valid graph by adding vertices one at a time."""
        A = start_A[:]
        n = start_n
        
        for _ in range(target_n - n):
            if time.time() > deadline - 5:
                break
            
            # Try adding a new vertex with random connections
            best_conn = None
            best_score = -1
            
            for attempt in range(20):
                # Randomly decide connections to existing vertices
                new_mask = 0
                for i in range(n):
                    if rng.random() < 0.5:
                        new_mask |= (1 << i)
                
                # Check if adding this vertex creates a K4 or ind6
                # K4 check: new vertex v connects to a,b,c where a,b,c form a triangle
                # ind6 check: new vertex v is non-adjacent to 5 vertices that form an independent set
                
                # For K4: for each triple (a,b,c) among neighbors of v that form a triangle,
                # we'd have K4 = {v,a,b,c}
                has_k4 = False
                neigh = []
                for i in range(n):
                    if (new_mask >> i) & 1:
                        neigh.append(i)
                
                for i in range(len(neigh)):
                    for j in range(i+1, len(neigh)):
                        a, b = neigh[i], neigh[j]
                        if (A[b] >> a) & 1:
                            for k in range(j+1, len(neigh)):
                                c = neigh[k]
                                if (A[c] >> a) & 1 and (A[c] >> b) & 1:
                                    has_k4 = True
                                    break
                    if has_k4:
                        break
                
                if has_k4:
                    continue
                
                # For ind6: new vertex is non-adjacent to some set S. If S contains an independent set of size 5,
                # then S ∪ {v} is an independent set of size 6.
                non_neigh = []
                for i in range(n):
                    if not ((new_mask >> i) & 1):
                        non_neigh.append(i)
                
                # Check if non_neigh contains an independent set of size 5
                # This is expensive; approximate by checking if any 5-tuple in non_neigh is independent
                # For efficiency, just check a few random subsets
                ind5_found = False
                if len(non_neigh) >= 5:
                    for _ in range(5):
                        subset = rng.sample(non_neigh, min(5, len(non_neigh)))
                        is_ind = True
                        for i in range(len(subset)):
                            for j in range(i+1, len(subset)):
                                if (A[subset[j]] >> subset[i]) & 1:
                                    is_ind = False
                                    break
                            if not is_ind:
                                break
                        if is_ind and len(subset) == 5:
                            ind5_found = True
                            break
                
                if ind5_found:
                    continue
                
                # Score: prefer connections that don't create near-violations
                score = len(neigh)  # Prefer more edges
                if score > best_score:
                    best_score = score
                    best_conn = new_mask
            
            if best_conn is not None:
                # Add the new vertex
                new_A = A[:]
                new_n = n + 1
                new_A.append(best_conn)
                for i in range(n):
                    if (best_conn >> i) & 1:
                        new_A[i] |= (1 << (n))
                A = new_A
                n = new_n
            else:
                # Can't add a vertex safely; break
                break
        
        return A[:n], n
    
    best_n = 0
    best_A = []
    
    # Try known constructions first
    # Paley graphs
    for q in [5, 13, 17, 29, 37]:
        if time.time() > deadline - 40:
            break
        A = make_paley(q)
        if A is not None:
            n = len(A)
            if valid(n, A):
                if n > best_n:
                    best_n = n
                    best_A = A[:]
    
    # Try circulant constructions for n=35, 36
    for target_n in [36, 35, 34]:
        if time.time() > deadline - 30:
            break
        
        # Generate many random difference sets
        rng = random.Random(42)
        
        for trial in range(200):
            if time.time() > deadline - 10:
                break
            
            # Random symmetric difference set
            D_half = set()
            # Choose roughly n/4 elements from 1..n//2
            num_elements = rng.randint(target_n // 5, target_n // 3)
            candidates = list(range(1, target_n // 2 + 1))
            rng.shuffle(candidates)
            D_half = set(candidates[:num_elements])
            
            A = make_cyclic_symmetric(target_n, D_half)
            
            # Quick check: degree should be reasonable
            deg = bin(A[0]).count('1')
            if deg < target_n // 3 or deg > 2 * target_n // 3:
                continue
            
            if valid(target_n, A):
                if target_n > best_n:
                    best_n = target_n
                    best_A = A[:]
                break
        
        if best_n >= target_n:
            break
    
    # Try local search from random starts
    if best_n < 35 and time.time() < deadline - 20:
        rng = random.Random(123)
        
        for target_n in [35, 36, 34, 33]:
            if time.time() > deadline - 10:
                break
            
            for restart in range(5):
                if time.time() > deadline - 10:
                    break
                
                # Start with a random graph with appropriate density
                p = 0.4  # Roughly the right density for K4-free and alpha<6
                A = random_adj(target_n, p, rng)
                A = try_local_search(target_n, A, rng, deadline, max_iters=200)
                
                if valid(target_n, A):
                    if target_n > best_n:
                        best_n = target_n
                        best_A = A[:]
                    break
            
            if best_n >= target_n:
                break
    
    # Try greedy growth from smaller valid graphs
    if best_n < 35 and time.time() < deadline - 15:
        rng = random.Random(777)
        
        # Start from a valid smaller graph and try to grow
        for start_n in [13, 17, 29]:
            if time.time() > deadline - 10:
                break
            
            A = make_paley(start_n)
            if A is None:
                continue
            
            if not valid(start_n, A):
                continue
            
            grown_A, grown_n = greedy_grow(A, start_n, min(36, start_n + 15), rng, deadline)
            
            if valid(grown_n, grown_A) and grown_n > best_n:
                best_n = grown_n
                best_A = grown_A[:]
            
            if best_n >= 35:
                break
    
    # Final attempt: more aggressive local search
    if best_n < 35 and time.time() < deadline - 10:
        rng = random.Random(999)
        
        for target_n in [35, 36]:
            if time.time() > deadline - 5:
                break
            
            for restart in range(10):
                if time.time() > deadline - 5:
                    break
                
                # Start with a structured graph
                # Use a random circulant as starting point
                D_half = set()
                num_elements = rng.randint(target_n // 4, target_n // 3)
                candidates = list(range(1, target_n // 2 + 1))
                rng.shuffle(candidates)
                D_half = set(candidates[:num_elements])
                
                A = make_cyclic_symmetric(target_n, D_half)
                A = try_local_search(target_n, A, rng, deadline, max_iters=300)
                
                if valid(target_n, A):
                    if target_n > best_n:
                        best_n = target_n
                        best_A = A[:]
                    break
            
            if best_n >= target_n:
                break
    
    # Final safety: if best_n is 0, return a trivial valid graph
    if best_n == 0:
        # Return C5
        n = 5
        A = [0] * n
        edges_c5 = [(0,1),(1,2),(2,3),(3,4),(4,0)]
        for u,v in edges_c5:
            A[u] |= (1<<v)
            A[v] |= (1<<u)
        best_n = 5
        best_A = A
    
    return {'n': best_n, 'edges': to_edges(best_n, best_A)}
