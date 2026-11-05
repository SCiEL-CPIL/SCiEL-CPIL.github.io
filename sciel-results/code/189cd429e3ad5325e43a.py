    def _synthesize_iswap():
        return [
            ["u3", 0, math.pi/2, 0, math.pi/2], # H
            ["u3", 1, math.pi/2, 0, math.pi/2], # H
            ["cx", 0, 1],
            ["u3", 0, math.pi/2, math.pi/2, 0], # ?
            ["u3", 1, math.pi/2, math.pi/2, 0],
            ["cx", 1, 0],
            ["u3", 0, math.pi/2, 0, math.pi/2], # H
            ["u3", 1, math.pi/2, 0, math.pi/2], # H
        ]
    