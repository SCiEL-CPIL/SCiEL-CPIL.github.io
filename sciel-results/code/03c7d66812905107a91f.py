# Forward
_update_autoconv(h, i, delta, C)
h[i] += delta

# Backward
_update_autoconv(h, i, -delta, C)
h[i] -= delta
