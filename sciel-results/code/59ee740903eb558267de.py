import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
        # bins is remaining capacity
        # We want to return scores. Higher score = better.
        # Valid bins: bins >= item
        
        valid_mask = bins >= item
        scores = np.full(len(bins), -np.inf)
        
        if not np.any(valid_mask):
            return scores
            
        valid_bins = bins[valid_mask]
        
        # Strategy: 
        # 1. Prefer bins that can fit at least 2 items of current size (R >= 2*item)
        # 2. Among those, minimize R - 2*item (best fit for 2 items)
        # 3. If none, prefer bins that fit 1 item (R >= item)
        # 4. Among those, minimize R - item (standard best fit)
        
        # Calculate residuals if we place item
        residuals = valid_bins - item
        
        # Condition: Can we fit another item of size 'item' in the residual?
        # residual >= item  =>  valid_bins - item >= item  => valid_bins >= 2 * item
        
        two_item_mask = valid_bins >= 2 * item
        
        if np.any(two_item_mask):
            # We have bins that can hold 2 items.
            # Score them by how close they are to 2*item.
            # We want to minimize (valid_bins - 2*item).
            # Score = -(valid_bins - 2*item)
            # This is high for small excess.
            
            # For invalid (1-item) bins, we give a very low score so they are not chosen.
            # But what if 2*item is huge and we have a bin with R=2.1*item and another with R=1.1*item?
            # The 1.1*item bin creates dead space 0.1*item.
            # The 2.1*item bin creates dead space 1.1*item (after placing 2 items? No, after placing 1).
            # Wait.
            # If we pick R=2.1*item, we place 1 item. R' = 1.1*item.
            # Next item (size item) fits. R'' = 0.1*item.
            # Total waste 0.1*item.
            
            # If we pick R=1.1*item, we place 1 item. R' = 0.1*item.
            # Next item (size item) does NOT fit.
            # Waste 0.1*item.
            
            # In this specific case, both result in 0.1*item waste.
            # But the 2-item bin might allow packing more items if the next one is smaller.
            # The 1-item bin is "dead" for items >= item.
            
            # So, generally, keeping capacity for the next item is better.
            
            # Let's score the 2-item bins by - (valid_bins - 2*item)
            # And the 1-item bins by a lower value.
            
            # To ensure 2-item bins are always preferred over 1-item bins,
            # we can add a large offset.
            
            # Max possible excess for 2-item bins?
            # If bins are large, excess is large.
            # Let's just use the raw negative excess for 2-item bins.
            # For 1-item bins, let's use - (valid_bins - item) - LARGE_PENALTY.
            
            # What is LARGE_PENALTY?
            # The max difference in (R-2*item) could be large.
            # Let's use a penalty that ensures any 2-item bin beats any 1-item bin.
            # Penalty = max(valid_bins) + 1?
            
            max_val = np.max(valid_bins)
            penalty = max_val + 1.0
            
            # Scores for 2-item bins
            scores_two = -(valid_bins[two_item_mask] - 2 * item)
            
            # Scores for 1-item bins
            one_item_mask = ~two_item_mask
            if np.any(one_item_mask):
                scores_one = -(valid_bins[one_item_mask] - item) - penalty
            
            # Assign to original scores array
            # We need to map back to original indices
            valid_indices = np.where(valid_mask)[0]
            
            # Create a local score array for valid bins
            local_scores = np.full(len(valid_bins), -np.inf)
            
            local_scores[two_item_mask] = scores_two
            if np.any(one_item_mask):
                local_scores[one_item_mask] = scores_one
            
            scores[valid_indices] = local_scores
            
        else:
            # No bins can hold 2 items. Fall back to Best Fit.
            # Minimize R - item.
            # Score = -(valid_bins - item)
            local_scores = -(valid_bins - item)
            
            valid_indices = np.where(valid_mask)[0]
            scores[valid_indices] = local_scores
            
        return scores
