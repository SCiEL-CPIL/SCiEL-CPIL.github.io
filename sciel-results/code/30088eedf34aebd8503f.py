    if best_blocks is None:
        # Emergency fallback: run greedy without time limit
        result, _ = run_greedy(None)
        if result is not None:
            best_blocks = [all_k_subsets[i] for i in result]
        else:
            return []
