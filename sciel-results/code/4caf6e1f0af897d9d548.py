<<<<<<< SEARCH
    # Try to improve the best layout with local optimization
    if best_inner is not None and time.time() < _DEADLINE - 20.0:
        # Find which layout was best
        for layout_idx, positions in enumerate(layouts):
            if time.time() > _DEADLINE - 20.0:
                break
            
            # Check if this layout matches the best
            match = True
            for i in range(N):
                if abs(positions[i][0] - best_inner[i][0]) > 1e-6 or abs(positions[i][1] - best_inner[i][1]) > 1e-6:
                    match = False
                    break
            if not match:
                continue
            
            # Local optimization
            x0 = []
            for i in range(N):
                x0.append(positions[i][0])
                x0.append(positions[i][1])
                x0.append(math.radians(inner_angle))
            
            # Try different outer angles
            for outer_angle_deg in [best_outer_angle, best_outer_angle + 5, best_outer_angle - 5]:
                if time.time() > _DEADLINE - 20.0:
                    break
                
                try:
                    res = minimize(_objective, x0, args=(positions, outer_angle_deg),
                                  method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8})
                    if res.fun < best_side and res.fun < 100:
                        new_positions = []
                        new_angles = []
                        for i in range(N):
                            new_positions.append((res.x[3*i], res.x[3*i+1]))
                            new_angles.append(math.degrees(res.x[3*i+2]))
                        
                        all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in zip(new_positions, new_angles)])
                        cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
                        
                        if side < best_side:
                            best_side = side
                            best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                            best_center = (cx, cy)
                            best_outer_angle = outer_angle_deg
                except Exception:
                    pass
            break
=======
    # Try to improve the best layout with local optimization
    if best_inner is not None and time.time() < _DEADLINE - 25.0:
        # Find which layout was best
        for layout_idx, positions in enumerate(layouts):
            if time.time() > _DEADLINE - 25.0:
                break
            
            # Check if this layout matches the best
            match = True
            for i in range(N):
                if abs(positions[i][0] - best_inner[i][0]) > 1e-6 or abs(positions[i][1] - best_inner[i][1]) > 1e-6:
                    match = False
                    break
            if not match:
                continue
            
            # Local optimization of positions and angles
            inner_angle = 30.0
            x0 = []
            for i in range(N):
                x0.append(positions[i][0])
                x0.append(positions[i][1])
                x0.append(math.radians(inner_angle))
            
            # Optimize positions for the best outer angle
            try:
                res = minimize(_objective, x0, args=(positions, best_outer_angle),
                              method='L-BFGS-B', options={'maxiter': 150, 'ftol': 1e-9})
                if res.fun < best_side and res.fun < 100:
                    new_positions = []
                    new_angles = []
                    for i in range(N):
                        new_positions.append((res.x[3*i], res.x[3*i+1]))
                        new_angles.append(math.degrees(res.x[3*i+2]))
                    
                    # Re-optimize outer angle with new positions
                    best_angle, best_c, best_s = _polish_outer_angle(
                        new_positions, new_angles, best_outer_angle, time_budget=5.0
                    )
                    
                    if best_s < best_side:
                        best_side = best_s
                        best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                        best_center = best_c
                        best_outer_angle = best_angle
            except Exception:
                pass
            break
>>>>>>> REPLACE
