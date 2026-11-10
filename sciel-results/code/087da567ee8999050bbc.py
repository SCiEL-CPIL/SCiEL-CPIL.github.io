import numpy as np
import random
from itertools import combinations

def construct():
    n = 20
    
    # Helper to check if a specific triple has >= 3 common neighbors
    def triple_violation_count(A, a, b, c):
        common = A[:, a] & A[:, b] & A[:, c]
        return np.sum(common)

    def find_violations(A):
        violations = []
        for a, b, c in combinations(range(n), 3):
            cnt = triple_violation_count(A, a, b, c)
            if cnt >= 3:
                violations.append((a, b, c))
        return violations

    def count_edges(A):
        return int(np.sum(A) // 2)

    def generate_polarity_graph_q4():
        # Generate Polarity Graph of PG(2,4) on 21 vertices
        # Field F4 = {0, 1, w, w2} where w^2 + w + 1 = 0
        # We represent elements as 0, 1, 2, 3
        # 0: 0
        # 1: 1
        # 2: w
        # 3: w^2
        # Addition table:
        # 0+0=0, 0+1=1, 0+2=2, 0+3=3
        # 1+1=0, 1+2=3, 1+3=2
        # 2+2=1, 2+3=0
        # 3+3=2
        add_table = np.array([
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 1, 0],
            [3, 2, 0, 1]
        ])
        # Multiplication table
        # 1 is identity
        # 2*2=1, 2*3=3? No. w*w = w^2 = 3. w*w^2 = 1. w^2*w^2 = w = 2.
        mul_table = np.array([
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ])
        
        # Points are non-zero vectors in F4^3, modulo scaling
        # There are (4^3 - 1) / (4 - 1) = 21 points
        # We can generate them by iterating all non-zero vectors and picking a representative
        # Representative: first non-zero component is 1
        
        points = []
        seen = set()
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize
                    if x != 0:
                        rep = (1, mul_table[x, y], mul_table[x, z]) # y/x, z/x
                        # Actually, we need to divide by x. In F4, x^-1 = x (since x^2=1 for x!=0,1? No. 1^-1=1, 2^-1=3, 3^-1=2)
                        # Let's just use the tuple (x,y,z) and identify (x,y,z) ~ (kx, ky, kz)
                        pass
                    
        # Simpler: Just list all 21 points manually or generate standard basis + others
        # Standard basis: (1,0,0), (0,1,0), (0,0,1)
        # (1,1,0), (1,1,1), (1,2,0), (1,2,1), (1,3,0), (1,3,1)
        # (0,1,1), (0,1,2), (0,1,3)
        # (0,0,1) is already there.
        # We need 21 points.
        
        # Let's use a known adjacency matrix for the Polarity Graph of PG(2,4) if possible?
        # No, let's construct it properly.
        
        # Vectors in F4^3
        vecs = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if x != 0 or y != 0 or z != 0:
                        vecs.append((x, y, z))
        
        # Group into lines (1D subspaces)
        lines = []
        used = set()
        for v in vecs:
            if v in used:
                continue
            # Generate all multiples
            subspace = []
            for k in range(4):
                if k == 0:
                    continue
                kv = (mul_table[k, v[0]], mul_table[k, v[1]], mul_table[k, v[2]])
                subspace.append(kv)
                used.add(kv)
            lines.append(subspace)
            
        # Now we have 21 lines (points of the dual projective plane, which is isomorphic)
        # Each line has 4 vectors.
        # Polarity: P ~ Q if sum(v_P * v_Q) = 0 for some/any representative?
        # The dot product is well-defined on the subspaces if the form is non-degenerate.
        
        def dot(u, v):
            return add_table[add_table[mul_table[u[0], v[0]], mul_table[u[1], v[1]]], mul_table[u[2], v[2]]]
            
        A = np.zeros((21, 21), dtype=bool)
        for i in range(21):
            for j in range(21):
                # Check if any vector in line i is orthogonal to any vector in line j?
                # Since it's a subspace, if one is orthogonal, all are?
                # u' = k u, v' = l v. u'.v' = kl (u.v).
                # So we just check one representative.
                rep_i = lines[i][0]
                rep_j = lines[j][0]
                if dot(rep_i, rep_j) == 0:
                    A[i, j] = True
                    A[j, i] = True
                    
        # Remove self loops (absolute points)
        np.fill_diagonal(A, False)
        
        # Remove one vertex to get n=20
        # Remove vertex 0
        A = A[1:, 1:]
        
        return A

    best_edges = []
    best_count = 0
    
    # Use the deterministic construction as the primary seed
    # Also try a few random seeds to be safe
    
    A0 = generate_polarity_graph_q4()
    
    candidates = [A0.copy()]
    
    # Add a couple of random dense graphs
    for seed in [42, 123]:
        random.seed(seed)
        np.random.seed(seed)
        A_rand = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.8:
                    A_rand[i, j] = True
                    A_rand[j, i] = True
        candidates.append(A_rand)
        
    for A in candidates:
        # Local Search
        max_iter = 2000
        for _ in range(max_iter):
            violations = find_violations(A)
            if not violations:
                break
            
            # Pick a random violation
            a, b, c = random.choice(violations)
            
            # Find common neighbors
            common = np.where(A[:, a] & A[:, b] & A[:, c])[0]
            if len(common) == 0:
                continue # Should not happen if it's a violation
            
            # Pick a random common neighbor x
            x = random.choice(common)
            
            # Remove edge (x, a)
            A[x, a] = False
            A[a, x] = False
            
            # Try to add a random missing edge to maintain density
            # Find missing edges
            missing = []
            for i in range(n):
                for j in range(i+1, n):
                    if not A[i, j]:
                        missing.append((i, j))
                        
            if missing:
                u, v = random.choice(missing)
                A[u, v] = True
                A[v, u] = True
                
                # Check if this new edge creates a violation
                # Only check triples involving u or v
                valid = True
                # Triples with u
                for b2, c2 in combinations([x for x in range(n) if x != u], 2):
                    if triple_violation_count(A, u, b2, c2) >= 3:
                        valid = False
                        break
                if valid:
                    # Triples with v (not u)
                    for b2, c2 in combinations([x for x in range(n) if x != v and x != u], 2):
                        if triple_violation_count(A, v, b2, c2) >= 3:
                            valid = False
                            break
                
                if not valid:
                    # Revert the addition
                    A[u, v] = False
                    A[v, u] = False
                    # Keep the removal of (x, a)
            
        # Final greedy add
        missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not A[i, j]]
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            A[u, v] = True
            A[v, u] = True
            valid = True
            for b2, c2 in combinations([x for x in range(n) if x != u], 2):
                if triple_violation_count(A, u, b2, c2) >= 3:
                    valid = False
                    break
            if valid:
                for b2, c2 in combinations([x for x in range(n) if x != v and x != u], 2):
                    if triple_violation_count(A, v, b2, c2) >= 3:
                        valid = False
                        break
            
            if not valid:
                A[u, v] = False
                A[v, u] = False

        current_count = count_edges(A)
        if current_count > best_count:
            best_count = current_count
            best_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if A[i, j]:
                        best_edges.append([i, j])

    return best_edges
