    # L-BFGS-B refinement on best
    if time.time() < deadline - 10 and len(best_materials) > 0:
        lbfgs_mats, lbfgs_thicks = _optimize_with_lbfgsb(
            list(best_materials), list(best_thicknesses), problem, wls, materials_dict,
            n_amb, n_sub, deadline - 5)
        if _is_valid(lbfgs_mats, lbfgs_thicks, problem):
            lbfgs_score = _evaluate(lbfgs_mats, lbfgs_thicks, wls, n_amb, n_sub, materials_dict)
            if lbfgs_score < best_score:
                best_score = lbfgs_score
                best_materials = list(lbfgs_mats)
                best_thicknesses = list(lbfgs_thicks)
