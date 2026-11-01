  def _high_precision_polish(cx, cy, r, W, H, time_budget):
      cx = cx.copy()
      cy = cy.copy()
      r = r.copy()
      deadline = time.time() + time_budget
      
      # Iteratively expand radii and adjust positions
      for _ in range(50):
          if time.time() > deadline: break
          
          # 1. Expand radii
          r_new = _expand_radii(cx, cy, W, H, deadline)
          if np.sum(r_new) > np.sum(r) + 1e-9:
              r = r_new
          
          # 2. Try to move circles to improve packing
          # For each circle, calculate a "pressure" from neighbors
          # Move towards the center of mass of the free space
          # Simple heuristic: move towards the center of the rectangle if it increases min distance
          moved = False
          for i in range(N):
              # Calculate current min distance
              min_dist = min(cx[i], W-cx[i], cy[i], H-cy[i])
              for j in range(N):
                  if i==j: continue
                  d = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
                  min_dist = min(min_dist, d - r[j])
              
              # Try small moves
              for dx in [-0.01, 0, 0.01]:
                  for dy in [-0.01, 0, 0.01]:
                      if dx==0 and dy==0: continue
                      new_cx = cx[i] + dx
                      new_cy = cy[i] + dy
                      # Check validity with current radii
                      if new_cx < r[i] or new_cx > W-r[i] or new_cy < r[i] or new_cy > H-r[i]:
                          continue
                      valid = True
                      for j in range(N):
                          if i==j: continue
                          d_sq = (new_cx-cx[j])**2 + (new_cy-cy[j])**2
                          if d_sq < (r[i]+r[j])**2 - 1e-9:
                              valid = False
                              break
                      if valid:
                          # Check if this move allows larger radius for i
                          # (Approx: check if min distance increased)
                          new_min_dist = min(new_cx, W-new_cx, new_cy, H-new_cy)
                          for j in range(N):
                              if i==j: continue
                              d = np.sqrt((new_cx-cx[j])**2 + (new_cy-cy[j])**2)
                              new_min_dist = min(new_min_dist, d - r[j])
                          
                          if new_min_dist > min_dist + 1e-6:
                              cx[i] = new_cx
                              cy[i] = new_cy
                              moved = True
                              break
                  if moved: break
              if moved: break
          
          if not moved:
              break
              
          # Re-expand
          r_new = _expand_radii(cx, cy, W, H, deadline)
          if np.sum(r_new) > np.sum(r):
              r = r_new
              
      return cx, cy, r
  