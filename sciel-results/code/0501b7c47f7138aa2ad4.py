import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    
    # Best-Fit strategy:
    # We want the bin with the SMALLEST remaining capacity among valid bins.
    # To achieve this with a "highest score wins" mechanism, we assign
    # scores inversely proportional to the remaining capacity.
    # score = -remaining
    # Smallest remaining -> Least negative (highest) score.
    # Largest remaining -> Most negative (lowest) score.
    
    # Invalid bins get -inf so they are never selected.
    scores = np.where(valid, -remaining, -np.inf)
    
    return scores
