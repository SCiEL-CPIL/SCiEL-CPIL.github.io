    import numpy as np

    def construct():
        # Inner hexagons: list of (x, y, angle_degrees)
        # Side length 1
        inner = [(0.0, 0.0, 0.0)]
        
        # Outer hexagon
        center = (0.0, 0.0)
        side = 1.0
        angle = 0.0
        
        return inner, center, side, angle
    