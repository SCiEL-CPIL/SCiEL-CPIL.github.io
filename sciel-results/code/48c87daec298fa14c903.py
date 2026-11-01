    else:
        # Safe default
        inner = [[k*sqrt3, 0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 20.0, 0.0
    