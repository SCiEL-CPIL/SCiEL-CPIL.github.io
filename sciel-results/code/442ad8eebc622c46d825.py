<<<<<<< SEARCH
        # Verify disjointness with tighter tolerance
        if check_all_disjoint(final_inner, eps=1e-4):
            if final_side < best_global_side:
                best_global_side = final_side
                best_global_inner = final_inner
                best_global_center = final_center
                best_global_angle = final_angle
        else:
            # Try to fix overlaps by slight expansion, then fall back
            fixed_inner = []
            for x, y, a in final_inner:
                fixed_inner.append([x, y, a])
            
            if check_all_disjoint(fixed_inner, eps=1e-5):
                if final_side < best_global_side:
                    best_global_side = final_side
                    best_global_inner = fixed_inner
                    best_global_center = final_center
                    best_global_angle = final_angle
            else:
                # Fall back to initial layout
                fallback_side = compute_required_side(initial_inner, (cx_avg, cy_avg), 0.0)
                if fallback_side < best_global_side:
                    best_global_side = fallback_side
                    best_global_inner = initial_inner
                    best_global_center = (cx_avg, cy_avg)
                    best_global_angle = 0.0
=======
        # Verify disjointness with tighter tolerance
        if check_all_disjoint(final_inner, eps=1e-4):
            if final_side < best_global_side:
                best_global_side = final_side
                best_global_inner = final_inner
                best_global_center = final_center
                best_global_angle = final_angle
        else:
            # Try to fix overlaps by slight expansion, then fall back
            fixed_inner = []
            for x, y, a in final_inner:
                fixed_inner.append([x, y, a])
            
            if check_all_disjoint(fixed_inner, eps=1e-5):
                if final_side < best_global_side:
                    best_global_side = final_side
                    best_global_inner = fixed_inner
                    best_global_center = final_center
                    best_global_angle = final_angle
            else:
                # Try a quick local fix: nudge overlapping hexagons apart
                fixed_inner2 = [list(h) for h in final_inner]
                for attempt in range(50):
                    overlap_found = False
                    for i in range(N):
                        vi = get_hex_vertices(fixed_inner2[i][0], fixed_inner2[i][1], 1.0, math.radians(fixed_inner2[i][2]))
                        for j in range(i + 1, N):
                            vj = get_hex_vertices(fixed_inner2[j][0], fixed_inner2[j][1], 1.0, math.radians(fixed_inner2[j][2]))
                            if sat_overlap_check(vi, vj, eps=1e-6):
                                overlap_found = True
                                # Nudge apart
                                dx = fixed_inner2[j][0] - fixed_inner2[i][0]
                                dy = fixed_inner2[j][1] - fixed_inner2[i][1]
                                d = math.sqrt(dx*dx + dy*dy) + 1e-9
                                nx = dx / d
                                ny = dy / d
                                fixed_inner2[i][0] -= 0.01 * nx
                                fixed_inner2[i][1] -= 0.01 * ny
                                fixed_inner2[j][0] += 0.01 * nx
                                fixed_inner2[j][1] += 0.01 * ny
                    if not overlap_found:
                        break

                if check_all_disjoint(fixed_inner2, eps=1e-5):
                    side_fixed2 = compute_required_side(fixed_inner2, final_center, final_angle)
                    if side_fixed2 < best_global_side:
                        best_global_side = side_fixed2
                        best_global_inner = fixed_inner2
                        best_global_center = final_center
                        best_global_angle = final_angle
                else:
                    # Fall back to initial layout
                    fallback_side = compute_required_side(initial_inner, (cx_avg, cy_avg), 0.0)
                    if fallback_side < best_global_side:
                        best_global_side = fallback_side
                        best_global_inner = initial_inner
                        best_global_center = (cx_avg, cy_avg)
                        best_global_angle = 0.0
>>>>>>> REPLACE
