    final_score, _, _, _, _ = get_score_details(best_points)
    if final_score <= 0:
        # Fallback to initial valid config if somehow we lost it
        # (Unlikely, but safe)
        return points # or re-init
