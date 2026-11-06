import numpy as np
from itertools import combinations

def construct():
    # We aim to construct a C4-free graph on n=47 vertices with min degree >= 8.
    # We use a Cayley graph on GF(49).
    # GF(49) is GF(7^2).
    # Let alpha be a root of x^2 + 1 = 0 in GF(7). 1^2=1, so x^2+1 = x^2-6.
    # Wait, 1^2 = 1 != -1 (which is 6).
    # Let's find an irreducible polynomial of degree 2 over GF(7).
    # x^2 + 3. Discriminant 0 - 12 = -12 = 2. 2 is not a square in GF(7) (squares: 1, 4, 2, 2, 4, 1).
    # Wait, 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1.
    # Squares: 1, 2, 4.
    # Non-squares: 3, 5, 6.
    # So x^2 + 3 is irreducible? Disc = -12 = 2. 2 is a square. So reducible.
    # x^2 + 1. Disc = -4 = 3. 3 is non-square. So x^2 + 1 is irreducible.
    # Let alpha^2 = -1 = 6.
    
    p = 7
    # Elements of GF(49) as pairs (a, b) representing a + b*alpha
    elements = [(a, b) for a in range(p) for b in range(p)]
    
    def add(x, y):
        return ((x[0] + y[0]) % p, (x[1] + y[1]) % p)
        
    def mul(x, y):
        a, b = x
        c, d = y
        # (a + b*alpha)(c + d*alpha) = ac + ad*alpha + bc*alpha + bd*alpha^2
        # alpha^2 = 6
        real = (a*c + 6*b*d) % p
        imag = (a*d + b*c) % p
        return (real, imag)
        
    def neg(x):
        return ((-x[0]) % p, (-x[1]) % p)
        
    # Precompute addition and negation
    add_table = {}
    neg_table = {}
    for i, x in enumerate(elements):
        neg_table[x] = neg(x)
        for j, y in enumerate(elements):
            add_table[(x, y)] = add(x, y)
            
    # Map elements to indices
    elem_index = {e: i for i, e in enumerate(elements)}
    index_elem = elements
    
    # We want to find a subset D of GF(49) such that the Cayley graph is C4-free.
    # To ensure C4-free, we can try to find a Sidon set S of size k, and set D = S U -S.
    # Or just search for D directly.
    # Given the time limit, we try a few random or structured D's.
    
    # A structured candidate: Multiplicative subgroup?
    # The multiplicative group has order 48.
    # Subgroups of size 4, 6, 8, 12, 16, 24.
    # If we take a subgroup H of size h, the graph with connection set H is C4-free?
    # No, if H is a subgroup, h1+h2+h3+h4=0 has many solutions.
    
    # Let's try to find a Sidon set of size 7.
    # We can use a greedy algorithm to find a Sidon set.
    
    def is_sidon(S):
        sums = {}
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = add(S[i], S[j])
                if s in sums:
                    return False
                sums[s] = (i, j)
        return True
        
    def find_sidon_set(size):
        # Greedy with backtracking or just random restarts
        import random
        random.seed(42)
        
        # Try to build a Sidon set of 'size'
        # We exclude 0 from S? If 0 in S, then 0+0=0.
        # Usually Sidon sets in groups exclude 0 or handle it carefully.
        # Let's pick from non-zero elements.
        candidates = [e for e in elements if e != (0,0)]
        
        for attempt in range(100):
            random.shuffle(candidates)
            S = []
            for e in candidates:
                if len(S) == size:
                    break
                # Check if adding e keeps it Sidon
                new_S = S + [e]
                if is_sidon(new_S):
                    S.append(e)
            if len(S) == size:
                return S
        return None
        
    S = find_sidon_set(7)
    
    if S is None:
        # Fallback to smaller or different
        S = find_sidon_set(6)
        
    if S is not None:
        D = set(S)
        for s in S:
            D.add(neg(s))
            
        # Check if Cayley graph with D is C4-free
        # C4 exists if there are d1, d2, d3, d4 in D such that d1+d2+d3+d4=0
        # and the vertices are distinct.
        # Vertices: x, x+d1, x+d1+d2, x+d1+d2+d3.
        # Distinctness requires partial sums to be non-zero and distinct.
        
        D_list = list(D)
        has_c4 = False
        for d1 in D_list:
            for d2 in D_list:
                if d1 == d2: continue # Simple C4 requires distinct edges? No, subgraph C4.
                # Actually, if d1=d2, then x, x+d1, x+2d1...
                # Let's just check the sum condition and distinctness.
                for d3 in D_list:
                    for d4 in D_list:
                        if add(add(add(d1, d2), d3), d4) == (0,0):
                            # Check distinct vertices
                            # x, x+d1, x+d1+d2, x+d1+d2+d3
                            # 1. d1 != 0
                            # 2. d1+d2 != 0
                            # 3. d1+d2+d3 != 0
                            # 4. d1+d2+d3+d4 == 0 (already checked)
                            # 5. d1 != d1+d2+d3+d4 (0) -> d1 != 0
                            # 6. d1+d2 != d1+d2+d3+d4 (0) -> d3+d4 != 0
                            # 7. d1+d2+d3 != d1+d2+d3+d4 (0) -> d4 != 0
                            # 8. d1 != d1+d2+d3 -> d2+d3 != 0
                            # 9. d1+d2 != d1+d2+d3 -> d3 != 0
                            # 10. d1+d2+d3 != d1+d2 -> d3 != 0
                            
                            if d1 != (0,0) and d2 != (0,0) and d3 != (0,0) and d4 != (0,0):
                                s12 = add(d1, d2)
                                s123 = add(s12, d3)
                                if s12 != (0,0) and s123 != (0,0):
                                    if add(d2, d3) != (0,0): # d1 != s123
                                        if add(d3, d4) != (0,0): # s12 != 0
                                            # We found a potential C4
                                            # We need to ensure the vertices are distinct in the graph
                                            # The condition above ensures x, x+d1, x+d1+d2, x+d1+d2+d3 are distinct.
                                            has_c4 = True
                                            break
                    if has_c4: break
                if has_c4: break
            if has_c4: break
            
        if not has_c4:
            # Construct graph on 49 vertices
            # We want n=47. Remove 2 vertices.
            # To maximize min degree, remove vertices with high degree?
            # No, all vertices have same degree in Cayley graph.
            # Degree = |D|.
            # If |D| is large enough, min degree in induced subgraph will be high.
            
            # Calculate degrees
            # In Cayley graph, deg(v) = |D|.
            # If we remove 2 vertices, min degree >= |D| - 2.
            # We need min degree >= 47 - 39 = 8.
            # So we need |D| - 2 >= 8 => |D| >= 10.
            # Since D = S U -S, and |S|=7, |D| is at most 14.
            # If S and -S are disjoint, |D|=14.
            
            if len(D) >= 10:
                # Select 47 vertices
                # Just take indices 0..46
                V = list(range(47))
                # Build edges
                edges = []
                for u in V:
                    eu = index_elem[u]
                    for v in V:
                        if u < v:
                            ev = index_elem[v]
                            diff = add(ev, neg(eu))
                            if diff in D:
                                edges.append([u, v])
                                
                # Verify min degree
                deg = [0]*47
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                    
                if min(deg) >= 8:
                    return {'n': 47, 'edges': edges}
                    
    # Fallback to current logic if this fails?
    # The directive says design a new program.
    # I should ensure this works or provide a robust fallback.
    # If the Sidon search fails, try random D of size 10-14.
    
    import random
    random.seed(42)
    non_zero = [e for e in elements if e != (0,0)]
    
    for _ in range(50):
        D = set(random.sample(non_zero, 12))
        # Symmetrize? No, just check C4.
        # If we don't symmetrize, graph is directed. We need undirected.
        # So we must use D such that d in D => -d in D.
        D_sym = set()
        for d in D:
            D_sym.add(d)
            D_sym.add(neg(d))
            
        # Check C4
        D_list = list(D_sym)
        has_c4 = False
        for d1 in D_list:
            for d2 in D_list:
                if d1 == d2: continue
                for d3 in D_list:
                    for d4 in D_list:
                        if add(add(add(d1, d2), d3), d4) == (0,0):
                            if d1 != (0,0) and d2 != (0,0) and d3 != (0,0) and d4 != (0,0):
                                s12 = add(d1, d2)
                                s123 = add(s12, d3)
                                if s12 != (0,0) and s123 != (0,0) and add(d2, d3) != (0,0) and add(d3, d4) != (0,0):
                                    has_c4 = True
                                    break
                    if has_c4: break
                if has_c4: break
            if has_c4: break
            
        if not has_c4 and len(D_sym) >= 10:
            V = list(range(47))
            edges = []
            for u in V:
                eu = index_elem[u]
                for v in V:
                    if u < v:
                        ev = index_elem[v]
                        diff = add(ev, neg(eu))
                        if diff in D_sym:
                            edges.append([u, v])
            deg = [0]*47
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if min(deg) >= 8:
                return {'n': 47, 'edges': edges}
                
    # If all fails, return a known valid small graph
    return {'n': 39, 'edges': []}
