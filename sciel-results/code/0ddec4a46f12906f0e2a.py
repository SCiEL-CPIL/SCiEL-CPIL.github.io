import numpy as np
lpf_arr = np.array(lpf)
# We need n in [10000, 1000000]
# lpf[n] is at index n, lpf[n+1] is at index n+1
# Range of n: 10000 to 1000000
# Corresponding lpf indices: 10000..1000000 and 10001..1000001
left = lpf_arr[10000:1000001]
right = lpf_arr[10001:1000002]
combined = np.maximum(left, right)
min_val = np.min(combined)
# Find the first occurrence of min_val
idx = np.argmin(combined)
best_n = 10000 + idx
