import numpy as np
import time

# Global state
_start_time = None

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _start_time
    
    # Initialize timing if not started
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 150s limit, keep 5s margin
    elapsed = time.time() - _start_time
    if elapsed > 145:
        # Fallback to simple Best Fit to ensure completion
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
        
    valid_indices = np.where(valid)[0]
    new_rems = remaining[valid]
    
    # Thermodynamic Model Parameters
    # We model the system as trying to reach a low-energy state.
    # The "energy" of a gap is high if it is "dead" (too small to be used) 
    # or if it is inefficiently large (wasted capacity that could have been filled).
    
    # Estimate the expected size of the next item.
    # For Weibull distributions in bin packing tasks with capacity 100,
    # items are often small. A common mean for Weibull(k=2, lambda~2-3) is around 2-5.
    # Let's assume a conservative expected next item size based on the current item size
    # or a global heuristic. 
    # If the current item is small, the next is likely small.
    # If the current item is large, the next could be anything, but statistically small.
    
    # Heuristic for Expected Next Item (ENI):
    # We use a smoothed estimate. If we don't have history, we assume a mean of ~3.0.
    # We can refine this: if the current item is very small (<2), assume next is ~1.5.
    # If current item is large (>10), assume next is ~3.0 (regression to mean).
    eni = 3.0
    if item < 2.0:
        eni = 1.5
    elif item < 5.0:
        eni = 2.5
    else:
        eni = 3.5
        
    # Define the "Optimal Gap" distribution.
    # In a thermodynamic sense, we want gaps to be either:
    # 1. Zero (perfect fit, lowest energy)
    # 2. Large enough to accept the next item (usable space)
    # Gaps in between (0 < gap < ENI) are "high energy" dead space.
    # Gaps > ENI are "lower energy" but have higher entropy (more freedom).
    # We want to minimize the "Free Energy" F = U - TS.
    # Here, U (Internal Energy) is penalized for dead space.
    # S (Entropy) favors spreading out? No, in packing, we want consolidation.
    # Let's stick to a potential energy function U(gap).
    
    # Potential Energy U(gap):
    # If gap < ENI: High penalty (dead space). The smaller the gap relative to ENI, the worse, 
    # UNLESS it's extremely close to 0 (perfect fit).
    # Actually, if gap is 0, it's perfect. If gap is 1 and ENI is 3, it's dead.
    # Let's define U(gap) as:
    # - If gap < ENI: Penalty increases as gap approaches ENI/2? No.
    #   Let's use a "Waste" metric. Waste = ENI - gap (if gap < ENI).
    #   But gap=0 is good. So we need a function that is low at 0, rises, and stays high until gap > ENI.
    #   Let's use: U(gap) = (ENI - gap)^2 for gap < ENI? 
    #   At gap=0: U=ENI^2. At gap=ENI: U=0. This prefers gap=ENI over gap=0.
    #   But gap=0 is better for packing density!
    
    # Let's rethink. The goal is to minimize the number of bins.
    # A gap of 0 is ideal. A gap of 1 is terrible if next item is 3.
    # A gap of 3 is ideal if next item is 3.
    # A gap of 10 is okay if next item is 3 (can fit, leaves 7).
    
    # Revised Thermodynamic Score:
    # We want to maximize "Stability".
    # Stability is high if:
    # 1. Gap is 0 (Perfect fit)
    # 2. Gap >= ENI (Usable space for next item)
    
    # Let's create a score that penalizes "Dead Space" (0 < gap < ENI).
    # Score = -Penalty(gap)
    
    # Penalty function:
    # If gap == 0: Penalty = 0 (Best)
    # If 0 < gap < ENI: Penalty is high. Max penalty at gap = ENI/2?
    #   Let's say Penalty = ENI - gap? No, that favors larger gaps within the dead zone.
    #   We want to avoid the dead zone entirely.
    #   If we must land in the dead zone, we prefer gap=0 (perfect fit) over gap=1.
    #   So, if gap < ENI:
    #       If gap is very small (< 0.5 * ENI), it's "almost perfect fit" -> Low penalty.
    #       If gap is close to ENI, it's "just too small" -> High penalty?
    #       Actually, if gap is just below ENI, it's useless. If gap is 0, it's perfect.
    #       So penalty should be high for (0, ENI) and low for 0.
    #       Let's use: Penalty = 1 if gap > 0 and gap < ENI. 
    #       But we need a continuous score.
    
    # Let's try a different angle: "Effective Capacity" from Inspiration 1.
    # Score = - |gap - Target| where Target is either 0 or > ENI.
    
    # Let's define two target regions:
    # Region A: gap == 0 (Perfect Fit)
    # Region B: gap >= ENI (Usable Fit)
    
    # We want to minimize distance to the union of these regions.
    # dist_to_perfect = gap
    # dist_to_usable = max(0, ENI - gap)
    
    # Combined Distance = min(dist_to_perfect, dist_to_usable)
    # If gap = 0: dist_to_perfect=0, dist_to_usable=ENI. min=0. Score=0.
    # If gap = 1, ENI=3: dist_to_perfect=1, dist_to_usable=2. min=1. Score=-1.
    # If gap = 2, ENI=3: dist_to_perfect=2, dist_to_usable=1. min=1. Score=-1.
    # If gap = 3, ENI=3: dist_to_perfect=3, dist_to_usable=0. min=0. Score=0.
    # If gap = 10, ENI=3: dist_to_perfect=10, dist_to_usable=0. min=0. Score=0.
    
    # This function creates a "valley" of good scores at gap=0 and gap>=ENI.
    # It penalizes the intermediate "dead zone".
    
    # However, between gap=3 and gap=10, the score is the same (0).
    # We prefer smaller gaps (consolidation) to save space for larger items later.
    # So, for gap >= ENI, we should prefer smaller gaps.
    # Let's add a small consolidation bias for gap >= ENI.
    
    # Refined Score:
    # 1. Calculate dist_to_perfect = gap
    # 2. Calculate dist_to_usable = max(0, ENI - gap)
    # 3. base_dist = min(dist_to_perfect, dist_to_usable)
    # 4. If gap >= ENI: apply consolidation bias. score = -base_dist - 0.01 * gap
    #    If gap < ENI: score = -base_dist
    
    # Let's check:
    # ENI = 3
    # gap=0: base_dist=0, gap<ENI. score=0.
    # gap=1: base_dist=min(1, 2)=1. score=-1.
    # gap=2: base_dist=min(2, 1)=1. score=-1.
    # gap=3: base_dist=min(3, 0)=0. gap>=ENI. score=0 - 0.03 = -0.03.
    # gap=4: base_dist=min(4, 0)=0. gap>=ENI. score=0 - 0.04 = -0.04.
    # gap=10: base_dist=0. gap>=ENI. score=-0.1.
    
    # This prefers gap=0 (score 0) > gap=3 (score -0.03) > gap=1 (score -1).
    # It strongly avoids the dead zone (1, 2).
    # It slightly prefers gap=3 over gap=10.
    
    # This looks like a solid thermodynamic-inspired heuristic.
    
    dist_to_perfect = new_rems
    dist_to_usable = np.maximum(0.0, eni - new_rems)
    base_dist = np.minimum(dist_to_perfect, dist_to_usable)
    
    scores_valid = -base_dist
    
    # Apply consolidation bias for usable gaps
    usable_mask = new_rems >= eni
    if np.any(usable_mask):
        scores_valid[usable_mask] -= 0.01 * new_rems[usable_mask]
        
    # Add a tiny index penalty for tie-breaking (lower index preferred)
    scores_valid -= np.arange(len(valid_indices)) * 1e-9
    
    scores[valid] = scores_valid
    
    return scores
