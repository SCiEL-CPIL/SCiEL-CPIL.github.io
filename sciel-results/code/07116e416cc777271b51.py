   max_proj = 0
   for k in range(6):
       nangle = np.radians(30 + k * 60)
       ...
       proj = abs(dx * nx + dy * ny)
       if proj > max_proj:
           max_proj = proj
   if max_proj > 1e-9:
       t = 1.0 / max_proj
   