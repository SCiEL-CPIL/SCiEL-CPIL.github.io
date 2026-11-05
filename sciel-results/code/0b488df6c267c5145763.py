   def construct():
       N = 64
       num_points = N * N
       
       # Precompute conflict masks
       # mask[u][v] is an int where bit w is 1 if w forms isosceles with u, v
       # Note: mask[u][v] == mask[v][u]
       
       # To save memory/time, we can compute on the fly in the inner loop?
       # No, precomputation is $N^3$ which is fast.
       
       # However, storing N^2 integers of N^2 bits is 16MB.
       # Python overhead for list of lists of ints might be high but manageable.
       
       # Let's try to optimize the inner loop.
       # Instead of updating F for every q in S, we can just check validity?
       # No, the F update is the efficient way.
       
       # Let's use a randomized greedy with restarts.
       
       best_set = set()
       
       # Precompute all conflict masks
       # This might take a few seconds.
       masks = [[0] * num_points for _ in range(num_points)]
       # Symmetric, so only compute u < v?
       # But accessing masks[u][v] is easier if full.
       
       # Actually, computing masks is O(N^4) if we iterate w for each u,v.
       # N=64, N^4 = 16M. 16M iterations in Python is ~1-2 seconds.
       # This is acceptable.
       
       # Optimization:
       # For each u, v:
       #   Calculate set of w.
       
       # Let's write the precomputation carefully.
       
       points = [(x, y) for y in range(N) for x in range(N)]
       # index = y * N + x
       
       # Precompute squared distances? No, just check conditions.
       
       # Condition 1: w is apex (wu = wv)
       # Condition 2: u is apex (uw = uv)
       # Condition 3: v is apex (vw = vu)
       
       # We can combine these.
       
       # To speed up, we can precompute for each center c and radius r, the list of points.
       # circles[c][r2] = list of points w such that dist2(c, w) == r2
       
       # And for bisectors?
       # Bisector of u, v is a line.
       
       # Let's stick to the O(N^4) precomputation if it's fast enough.
       # 16M iterations.
       # Inside: check 3 conditions.
       # Condition 1: (x_w-x_u)^2 + (y_w-y_u)^2 == (x_w-x_v)^2 + (y_w-y_v)^2
       # Condition 2: (x_w-x_u)^2 + (y_w-y_u)^2 == (x_u-x_v)^2 + (y_u-y_v)^2
       # Condition 3: (x_w-x_v)^2 + (y_w-y_v)^2 == (x_u-x_v)^2 + (y_u-y_v)^2
       
       # This is very fast per iteration.
       
       # Then Greedy:
       # Shuffle indices
       # F = 0 (bitset)
       # S = []
       # for p in shuffled_indices:
       #   if not (F >> p) & 1:
       #     S.append(p)
       #     for q in S[:-1]:
       #       F |= masks[p][q]
       
       # This inner loop `for q in S` is the bottleneck.
       # If |S| is large, this is slow.
       # But |S| is likely small relative to N^2.
       # If |S| ~ 100, and we do 100 ORs of 4096-bit ints, that's fast.
       
       # We can run this greedy multiple times (e.g., 100 times) and keep the best.
       
   