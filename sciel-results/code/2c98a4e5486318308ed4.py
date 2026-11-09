import numpy as np
n, m, p = 4, 4, 5
rank = 49 + 16
U = np.zeros((n * m, rank))
# ... fills ...
return {"U": U, "V": V, "W": W}
