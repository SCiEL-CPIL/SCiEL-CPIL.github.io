import numpy as np
from autoresearch_suite.chem_clusters import relax

def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    def get_icosahedron():
        phi = (1 + np.sqrt(5)) / 2
        v = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        return v * 0.5

    pts = [np.zeros(3)]
    ico1 = get_icosahedron()
    pts.extend(ico1)
    
    if n > 13:
        remaining = n - len(pts)
        # Deterministic Fibonacci sphere for outer shell
        # Radius chosen to be outside shell 1 (R~0.95) but close enough for relaxation
        radius = 1.8
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(remaining):
            z = 1 - (i + 0.5) * 2 / remaining
            r = np.sqrt(1 - z*z)
            theta = 2 * np.pi * i / golden_ratio
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            pts.append([radius*x, radius*y, radius*z])
            
    start = np.array(pts)
    energy, positions = relax(start, 'morse', rho)
    return positions
