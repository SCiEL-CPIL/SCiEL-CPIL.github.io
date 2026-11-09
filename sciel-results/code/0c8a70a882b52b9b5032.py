            pos = res.x[:2*N].reshape(N, 2)
            ang = res.x[2*N:]
            pen = _penalty(pos, ang)
            if pen < 1e-4:
                side, center = _calculate_required_side_and_center(pos, ang, 0.0)
                if side < best_side:
                    # update best
