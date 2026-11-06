import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # We will construct the polarity graph of PG(2, q) for q=9.
    # If it fails or takes too long, we fall back to q=7.
    
    def build_polarity_graph(q):
        # q must be a prime power
        # For simplicity, we handle q=9 (F_9) and q=7 (F_7)
        
        if q == 9:
            # F_9 = F_3[i] / (i^2+1)
            # Elements are (a, b) representing a + bi
            def add(u, v):
                return ((u[0] + v[0]) % 3, (u[1] + v[1]) % 3)
            
            def mul(u, v):
                a, b = u
                c, d = v
                return ((a*c - b*d) % 3, (a*d + b*c) % 3)
            
            def sub(u, v):
                return ((u[0] - v[0]) % 3, (u[1] - v[1]) % 3)
            
            def neg(u):
                return ((-u[0]) % 3, (-u[1]) % 3)
            
            def eq(u, v):
                return u[0] == v[0] and u[1] == v[1]
            
            def is_zero(u):
                return u[0] == 0 and u[1] == 0
            
            zero = (0, 0)
            one = (1, 0)
            
        elif q == 7:
            # F_7
            def add(u, v): return (u + v) % 7
            def mul(u, v): return (u * v) % 7
            def sub(u, v): return (u - v) % 7
            def neg(u): return (-u) % 7
            def eq(u, v): return u == v
            def is_zero(u): return u == 0
            zero = 0
            one = 1
        else:
            raise ValueError("q must be 7 or 9")
            
        # Enumerate points in PG(2, q)
        # Points are [x:y:z] with (x,y,z) != (0,0,0), modulo scaling
        # We fix z=1 for affine points (q^2 points)
        # And z=0 for points at infinity (q+1 points)
        
        points = []
        
        # Affine points: z=1
        for x in range(q if q==7 else 3):
            for y in range(q if q==7 else 3):
                if q == 9:
                    # x, y are in F_3, so we iterate 0,1,2
                    # But wait, F_9 has 9 elements.
                    # My loop range is wrong for q=9 if I just use range(q).
                    # For q=9, elements are pairs.
                    pass
                    
        # Let's refactor to handle F_9 elements explicitly.
        if q == 9:
            F = [(a, b) for a in range(3) for b in range(3)]
            # Map element to index for lookup? Or just use tuples.
            # Points:
            # 1. z=1: x, y in F. 81 points.
            for x in F:
                for y in F:
                    points.append((x, y, (1, 0)))
            # 2. z=0: x, y in F, not both 0. 8 points? No, 9-1=8.
            # Wait, line at infinity has q+1 = 10 points.
            # In projective coords [x:y:0], (x,y) != (0,0).
            # Pairs (x,y) in F^2 \ {0}. Total 8.
            # But we need to mod out by scaling.
            # For z=0, scaling factor is in F*.
            # We can fix x=1 if x!=0, else y=1.
            # If x=1, y can be any of 9 values. (9 points)
            # If x=0, y must be 1 (since y!=0). (1 point)
            # Total 10.
            for y in F:
                points.append(((1, 0), y, (0, 0)))
            points.append(((0, 0), (1, 0), (0, 0)))
            
        elif q == 7:
            F = list(range(7))
            for x in F:
                for y in F:
                    points.append((x, y, 1))
            for y in F:
                points.append((1, y, 0))
            points.append((0, 1, 0))
            
        n = len(points)
        # n should be q^2 + q + 1
        
        edges = []
        
        # Bilinear form B(u, v) = u0v0 + u1v1 + u2v2
        def B(u, v):
            return add(mul(u[0], v[0]), add(mul(u[1], v[1]), mul(u[2], v[2])))
            
        for i in range(n):
            for j in range(i + 1, n):
                if is_zero(B(points[i], points[j])):
                    edges.append([i, j])
                    
        return {'n': n, 'edges': edges}
        
    # Try q=9 first
    try:
        result = build_polarity_graph(9)
        # Quick check: n should be 91
        if result['n'] != 91:
             raise ValueError("Wrong n")
        return result
    except Exception as e:
        # Fallback to q=7
        try:
            result = build_polarity_graph(7)
            if result['n'] != 57:
                raise ValueError("Wrong n")
            return result
        except Exception:
            # Fallback to q=5 (n=31)
            # Implementing q=5 is similar
            # For safety, return the seed solution
            edges = []
            for t in range(10):
                a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
                edges += [[a, b], [a, c], [b, c]]
            return {'n': 30, 'edges': edges}

