import numpy as np

def erdos_szekeres_problem7() -> np.ndarray:
    n = 33
    # Generate x coordinates evenly spaced
    x = np.linspace(0.1, 0.9, n)
    
    # Generate y coordinates using a sine wave
    # Use a frequency that creates multiple oscillations
    # 2 full periods in [0, 1]
    y = 0.5 + 0.4 * np.sin(4 * np.pi * x)
    
    points = np.column_stack((x, y))
    
    # Add tiny perturbation to ensure general position and avoid any accidental symmetries
    # although the sine wave is already in general position for most triples
    rng = np.random.default_rng(42)
    perturbation = rng.uniform(-1e-5, 1e-5, size=points.shape)
    points += perturbation
    
    # Clamp to [0, 1] just in case, though 0.1..0.9 +/- 1e-5 is safe
    points = np.clip(points, 0.0, 1.0)
    
    return points
