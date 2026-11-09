import numpy as np

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
       p = 1.0 / (2.0 * np.sqrt(2.0))
       support = np.array([0.0, 1000.0], dtype=np.float64)
       weights = np.array([1.0 - p, p], dtype=np.float64)
       return support, weights
