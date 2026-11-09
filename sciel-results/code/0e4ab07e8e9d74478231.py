<<<<<<< SEARCH
    # Final polish on the best result if we have time
    if best_points is not None and time.time() < deadline - 3.0:
        best_points = _polish(best_points, max_iters=500, deadline=deadline)
=======
    # Final polish on the best result if we have time
    if best_points is not None and time.time() < deadline - 5.0:
        best_points = _polish(best_points, max_iters=500, deadline=deadline)
    
    # Final projected gradient ascent if time allows
    if best_points is not None and time.time() < deadline - 3.0:
        remaining_time = deadline - time.time() - 2.0
        if remaining_time > 3.0:
            pg_final_iters = min(300, int(remaining_time * 10))
            best_points = _projected_gradient_ascent(best_points, max_iters=pg_final_iters, deadline=deadline)
>>>>>>> REPLACE
