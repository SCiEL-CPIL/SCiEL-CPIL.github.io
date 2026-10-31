import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # First-Fit strategy:
    # We want the first bin (lowest index) that has enough remaining capacity.
    # The input 'bins' contains the remaining capacity of each bin.
    
    # A bin is valid if its remaining capacity is >= item.
    valid = bins >= item
    
    # Assign a constant score (0.0) to all valid bins and -inf to invalid ones.
    # np.argmax (used by the evaluator to select the bin) returns the index of
    # the first occurrence of the maximum value. Thus, this implements First-Fit.
    scores = np.where(valid, 0.0, -np.inf)
    
    return scores
