spec = np.zeros(K_spec, dtype=complex)
# 0 and N/2 are real
spec[0] = mag[0] * np.exp(1j * phases[0]) # phases[0] should be 0 or pi for real? 
# Actually, if we want real IFFT, spec[0] and spec[N/2] must be real.
# So we should force phases[0]=0 and phases[N/2]=0 (or pi).
# Or just take real part of the product?
# Easier:
spec[0] = mag[0]
spec[N//2] = mag[N//2]
for k in range(1, N//2):
    spec[k] = mag[k] * np.exp(1j * phases[k])
    spec[N-k] = np.conj(spec[k])
