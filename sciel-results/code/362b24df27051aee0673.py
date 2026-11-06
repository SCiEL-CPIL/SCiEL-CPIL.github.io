<<<<<<< SEARCH
    # Final polish: one more local search pass
    if time.time() < deadline - 2:
        best, final_score = local_search(best, max_iters=100)
        if final_score > best_score:
            best_score = final_score
=======
    # Final polish: one more local search pass with more iterations
    if time.time() < deadline - 2:
        best, final_score = local_search(best, max_iters=200)
        if final_score > best_score:
            best_score = final_score
>>>>>>> REPLACE

