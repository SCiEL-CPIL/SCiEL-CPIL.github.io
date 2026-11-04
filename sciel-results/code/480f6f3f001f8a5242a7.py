   def min_hexagon_side(points, center, rotation):
       # rotation is in radians
       # Outer normals
       max_apothem = 0
       for k in range(6):
           angle = rotation + np.radians(60 * k)
           nx = np.cos(angle)
           ny = np.sin(angle)
           
           max_proj = -float('inf')
           min_proj = float('inf')
           
           for (px, py) in points:
               # Support of inner hexagon in direction (nx, ny)
               # Inner hexagon rotation is 30 deg (pi/6)
               # Side normals of inner hex are at pi/6 + 0, pi/6 + pi/3, ...?
               # Wait, earlier I said if vertex is at 30, normal is at 0.
               # Let's re-verify.
               # Vertex at 30. Side connects 30 and 90? No, vertices are 30, 90, 150...
               # Side 1: 30 to 90. Midpoint angle 60. Normal 60.
               # Side 2: 90 to 150. Midpoint 120. Normal 120.
               # ...
               # Side 6: 330 to 30. Midpoint 0. Normal 0.
               # So normals are 0, 60, 120, 180, 240, 300.
               
               # Angle of direction (nx, ny)
               theta = np.arctan2(ny, nx)
               
               # Find angle difference to nearest normal (0, 60, ...)
               # Normalize theta to [0, 360)
               theta_deg = np.degrees(theta) % 360
               # Nearest multiple of 60
               nearest = round(theta_deg / 60) * 60
               diff = abs(theta_deg - nearest)
               if diff > 30: diff = 60 - diff # just in case, but round handles it mostly
               
               # Support h
               h = (np.sqrt(3)/2) / np.cos(np.radians(diff))
               
               proj = px * nx + py * ny
               
               # The extent in direction n is [proj - h, proj + h]
               max_proj = max(max_proj, proj + h)
               min_proj = min(min_proj, proj - h)
           
           # Outer hexagon constraint for this normal
           # Center is (cx, cy)
           # Projection of center
           c_proj = center[0] * nx + center[1] * ny
           
           # We need c_proj + A >= max_proj  => A >= max_proj - c_proj
           # And c_proj - A <= min_proj  => A >= c_proj - min_proj
           
           req_A1 = max_proj - c_proj
           req_A2 = c_proj - min_proj
           
           max_apothem = max(max_apothem, req_A1, req_A2)
       
       return 2.0 * max_apothem / np.sqrt(3)
   