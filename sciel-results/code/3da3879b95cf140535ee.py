        if best_idx != -1:
            new_block = candidates[best_idx]
            # Check if already in blocks
            # For efficiency, we can skip this check if we are confident random sampling avoids it, 
            # but for safety:
            if new_block not in blocks:
                blocks.append(new_block)
                uncovered &= ~masks[best_idx]
                candidates.pop(best_idx)
                masks.pop(best_idx)
            else:
                # If duplicate, just remove from candidates and continue
                candidates.pop(best_idx)
                masks.pop(best_idx)
