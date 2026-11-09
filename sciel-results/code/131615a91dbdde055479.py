import numpy as np
# is_pw = np.zeros(N_MAX+1, dtype=np.int32)
# for p in powerful: is_pw[p] = 1
# fft_pw = np.fft.fft(is_pw, n=2*(N_MAX+1))
# corr = np.real(np.fft.ifft(fft_pw * np.conj(fft_pw)))
# corr gives count of ordered pairs (a,b) with a+b = k
# For a<=b: count = (corr[N] + (1 if N%2==0 and is_pw[N//2] else 0)) // 2
