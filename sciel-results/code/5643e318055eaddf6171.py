<<<<<<< SEARCH
    if len(candidates) > 1:
        # Use majority vote to select the consensus tree
        all_splits = [c[1] for c in candidates]
        consensus_splits = _majority_vote(all_splits)
        
        # Find the candidate closest to the consensus
        best_idx = 0
        best_score = float('inf')
        for i, (tree_i, splits_i) in enumerate(candidates):
            # RF distance to consensus
            score = _rf_distance(splits_i, consensus_splits)
            if score < best_score:
                best_score = score
                best_idx = i
        
        best_tree = candidates[best_idx][0]
        
        # Optional: NNI Refinement on the best tree
        # best_tree = _nni_refine(best_tree, d_full, labels)
        
        return best_tree + ";"
    else:
        return tree_full + ";"
=======
    if len(candidates) > 1:
        # Use majority vote to select the consensus tree
        all_splits = [c[1] for c in candidates]
        consensus_splits = _majority_vote(all_splits)
        
        # Find the candidate closest to the consensus
        best_idx = 0
        best_score = float('inf')
        for i, (tree_i, splits_i) in enumerate(candidates):
            # RF distance to consensus
            score = _rf_distance(splits_i, consensus_splits)
            if score < best_score:
                best_score = score
                best_idx = i
        
        best_tree = candidates[best_idx][0]
        
        # NNI Refinement on the best tree using likelihood-based scoring
        if _time_left() > 5.0:
            best_tree = _nni_refine_likelihood(best_tree, arr, labels)
        
        return best_tree + ";"
    else:
        return tree_full + ";"
>>>>>>> REPLACE

