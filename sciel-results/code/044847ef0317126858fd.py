def _writable_zeros(shape):
    a = np.zeros(shape)
    try:
        a.flags.writeable = True
    except ValueError:
        a = a.copy()
        a.flags.writeable = True
    return a
