<<<<<<< SEARCH
        new_score = full_score(cfg, tuple(center), angle)
        
        if new_score is not None:
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
            else:
                # Revert
                if r < 0.50:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.70:
                    cfg[idx][2] = old_ang
                elif r < 0.80:
                    center[0], center[1] = old_cx, old_cy
                elif r < 0.90:
                    angle = old_angle
                else:
                    cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                    cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
        else:
            # Invalid: revert
            if r < 0.50:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.70:
                cfg[idx][2] = old_ang
            elif r < 0.80:
                center[0], center[1] = old_cx, old_cy
            elif r < 0.90:
                angle = old_angle
            else:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
=======
        new_soft_score, new_depth = full_score_soft(cfg, tuple(center), angle)
        
        # Acceptance criterion based on soft score
        if new_soft_score < cur_score or (T > T_end and random.random() < math.exp(-(new_soft_score - cur_score) / T)):
            cur_score = new_soft_score
            # Update best valid if this state is valid
            if new_depth < 1e-6:
                valid_side = _outer_hex_side(cfg, tuple(center), angle)
                if valid_side < best_valid_score:
                    best_valid_score = valid_side
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
        else:
            # Revert
            if r < 0.50:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.70:
                cfg[idx][2] = old_ang
            elif r < 0.80:
                center[0], center[1] = old_cx, old_cy
            elif r < 0.90:
                angle = old_angle
            else:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
>>>>>>> REPLACE
