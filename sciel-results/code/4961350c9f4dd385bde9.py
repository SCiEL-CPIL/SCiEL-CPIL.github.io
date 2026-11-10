def _local_search(S, max_iters=2000):
    S = list(S)
    S_set = set(S)
    full = list(range(1, 129))
    best_size = len(S)
    
    for _ in range(max_iters):
        if not _within_deadline():
            break
        
        improved = False
        
        # 1. Try to add elements
        # We do a full pass to ensure we can't add anything
        added_any = True
        while added_any:
            added_any = False
            for x in full:
                if x in S_set:
                    continue
                if _can_add(x, S_set):
                    S.append(x)
                    S_set.add(x)
                    added_any = True
                    if len(S) > best_size:
                        best_size = len(S)
                    break # Restart the loop to pick up new opportunities? 
                    # Actually, just continuing the for loop is faster and sufficient for greedy.
                    # But restarting ensures we don't miss things if order matters? 
                    # Order doesn't matter for "can add". 
                    # So just continue the for loop.
                    pass
            # If we used 'break', we restart. If we don't, we finish the pass.
            # Let's finish the pass for efficiency.
        
        # If we added anything in the pass, we continue the outer loop to try again
        # (though the pass already exhausted all addable elements)
        # Wait, if I don't break on add, I might add multiple elements in one pass.
        # That's good.
        
        # If added_any was True, we improved. Continue to next iteration to check again?
        # No, if we exhausted all addable elements, we are stuck.
        # So if added_any is False, we proceed to kicks.
        
        if not added_any:
            # 2. Try kicks
            kick_improved = False
            # Try a few random kicks
            for _ in range(5):
                if not S:
                    break
                # Pick a random element to remove
                idx = random.randint(0, len(S) - 1)
                removed = S.pop(idx)
                S_set.discard(removed)
                
                # Greedily add back
                old_size = len(S)
                # Try to add all possible
                # We can use the same logic as above
                add_improved = True
                while add_improved:
                    add_improved = False
                    for x in full:
                        if x in S_set:
                            continue
                        if _can_add(x, S_set):
                            S.append(x)
                            S_set.add(x)
                            add_improved = True
                            break
                    # If we break on first add, we loop again.
                    # This is O(N*K).
                
                if len(S) > best_size:
                    best_size = len(S)
                    kick_improved = True
                    improved = True
                    break # Accept this kick and go to next outer iteration
                else:
                    # Revert
                    S.append(removed)
                    S_set.add(removed)
            
            if not kick_improved:
                # No kick improved the set. 
                # We are in a local minimum.
                # Break to allow a new restart in the main loop?
                # Or try a more aggressive kick?
                # For now, break.
                break
                
    return S
