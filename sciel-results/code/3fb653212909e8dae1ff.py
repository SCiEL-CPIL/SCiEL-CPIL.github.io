import itertools

def construct():
    n = 20
    
    # Construct PG(2,4) polarity graph
    # Field F4 = {0, 1, 2, 3} where 2 is alpha, 3 is alpha+1
    # alpha^2 = alpha + 1 => 2*2 = 3
    # 2*3 = 2*(2+1) = 4+2 = 1+2 = 3? No.
    # Let's define multiplication table for F4.
    # 0: 0,0,0,0
    # 1: 0,1,2,3
    # 2: 0,2,3,1 (2*2=3, 2*3=2*(2+1)=3+2=1)
    # 3: 0,3,1,2 (3*2=1, 3*3=(2+1)(2+1)=4+2+2+1=1+1+1+1=0? No. (2+1)^2 = 4+2+1 = 1+2+1 = 2? 
    # Let's use a standard representation.
    # F4 = GF(2)[x] / (x^2+x+1)
    # Elements: 0, 1, x, x+1
    # x^2 = x+1
    # (x+1)^2 = x^2+1 = x+1+1 = x
    # So:
    # 0: 0
    # 1: 1
    # 2: x
    # 3: x+1
    # Mult:
    # 2*2 = x^2 = 3
    # 2*3 = x(x+1) = x^2+x = (x+1)+x = 1
    # 3*3 = (x+1)^2 = x
    # So table:
    #   0 1 2 3
    # 0 0 0 0 0
    # 1 0 1 2 3
    # 2 0 2 3 1
    # 3 0 3 1 2
    
    def f4_mult(a, b):
        table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        return table[a][b]
        
    def f4_add(a, b):
        return a ^ b # XOR for GF(2) based fields
        
    # Points of PG(2,4) are non-zero vectors in F4^3 up to scaling.
    # Total non-zero vectors: 4^3 - 1 = 63.
    # Each line has 4 non-zero vectors (scalar multiples).
    # 63 / 3 = 21 points.
    
    points = []
    seen = set()
    for x in range(4):
        for y in range(4):
            for z in range(4):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonicalize: make first non-zero element 1
                vec = [x, y, z]
                idx = -1
                for i in range(3):
                    if vec[i] != 0:
                        idx = i
                        break
                # Multiply by inverse of vec[idx] to make it 1
                # Inverse in F4: 1->1, 2->3, 3->2
                inv_map = {1: 1, 2: 3, 3: 2}
                inv = inv_map[vec[idx]]
                canon = [f4_mult(v, inv) for v in vec]
                key = tuple(canon)
                if key not in seen:
                    seen.add(key)
                    points.append(canon)
                    
    # points has 21 elements.
    # Define polarity: x . y = x0*y0 + x1*y1 + x2*y2
    # Adjacent if dot product is 0.
    
    def dot_product(v1, v2):
        return f4_add(f4_add(f4_mult(v1[0], v2[0]), f4_mult(v1[1], v2[1])), f4_mult(v1[2], v2[2]))
        
    adj_21 = [[0]*21 for _ in range(21)]
    for i in range(21):
        for j in range(i+1, 21):
            if dot_product(points[i], points[j]) == 0:
                adj_21[i][j] = 1
                adj_21[j][i] = 1
                
    # Remove one vertex to get 20.
    # Remove vertex 0 (arbitrary).
    # Re-index remaining vertices to 0..19.
    adj = [[0]*n for _ in range(n)]
    mapping = [i for i in range(21) if i != 0]
    # mapping[k] is the index in the 21-graph of the k-th vertex in the 20-graph.
    # Wait, mapping is a list of 20 indices from 0..20.
    for i in range(n):
        for j in range(n):
            adj[i][j] = adj_21[mapping[i]][mapping[j]]
            
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append((i, j))
                
    # Local search: try to add edges
    def is_k33_free(adj_mat):
        n_local = len(adj_mat)
        for u, v, w in itertools.combinations(range(n_local), 3):
            common = 0
            for x in range(n_local):
                if adj_mat[u][x] and adj_mat[v][x] and adj_mat[w][x]:
                    common += 1
            if common > 2:
                return False
        return True
        
    # Greedy addition
    improved = True
    while improved:
        improved = False
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    # Try adding
                    adj[u][v] = 1
                    adj[v][u] = 1
                    if is_k33_free(adj):
                        edges.append((u, v))
                        improved = True
                    else:
                        adj[u][v] = 0
                        adj[v][u] = 0
                        
    res = [[u, v] for u, v in edges]
    return res
