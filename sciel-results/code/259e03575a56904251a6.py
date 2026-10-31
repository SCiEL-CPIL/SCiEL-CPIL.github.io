   for a in range(p):
       for b in range(p):
           u_idx = idx(a, b, 0)
           # ... solve for c, d ...
           v_idx = idx(c, d, 1)
           edges.append((u_idx, v_idx))
   