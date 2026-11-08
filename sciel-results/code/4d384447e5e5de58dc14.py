        if best_obj < 1e-6:
            U_snap, V_snap, W_snap = snap_to_rational(best_U, best_V, best_W)
            obj_snap = objective_smooth(U_snap, V_snap, W_snap, n, m, p, rank)
            if obj_snap < 1e-9:
                U_list = [[round(x, 4) for x in row] for row in U_snap]
                V_list = [[round(x, 4) for x in row] for row in V_snap]
                W_list = [[round(x, 4) for x in row] for row in W_snap]
                return {"U": U_list, "V": V_list, "W": W_list, "rank": rank}
