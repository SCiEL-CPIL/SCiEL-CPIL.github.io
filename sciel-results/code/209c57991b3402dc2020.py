<<<<<<< SEARCH
    # Robustly locate the data dictionary P
    P = None
    # 1. Check current module globals
    if 'P' in globals():
        P = globals()['P']
    # 2. Check __main__
    if P is None and hasattr(__main__, 'P'):
        P = getattr(__main__, 'P')
    # 3. Search all loaded modules
    if P is None:
        for mod in sys.modules.values():
            if hasattr(mod, 'P'):
                candidate = getattr(mod, 'P')
                # Verify it looks like the expected data structure
                if isinstance(candidate, dict) and 'edge_order' in candidate:
                    P = candidate
                    break
    # 4. Fallback to other common names in __main__
    if P is None:
        for name in ['params', 'data', 'config']:
            if hasattr(__main__, name):
                candidate = getattr(__main__, name)
                if isinstance(candidate, dict) and 'edge_order' in candidate:
                    P = candidate
                    break

    if P is None:
        raise NameError("Could not find data dictionary P")
=======
    # Robustly locate the data dictionary P
    P = None
    
    def find_data_dict(d):
        if isinstance(d, dict) and 'edge_order' in d:
            return d
        return None

    # 1. Scan current module globals for any dict with 'edge_order'
    for name, val in globals().items():
        if find_data_dict(val):
            P = val
            break
            
    # 2. Scan __main__ globals for any dict with 'edge_order'
    if P is None:
        for name, val in vars(__main__).items():
            if find_data_dict(val):
                P = val
                break
                
    # 3. Search all loaded modules for any dict with 'edge_order'
    if P is None:
        for mod in sys.modules.values():
            try:
                mod_vars = vars(mod)
                for name, val in mod_vars.items():
                    if find_data_dict(val):
                        P = val
                        break
            except Exception:
                continue
            if P is not None:
                break

    if P is None:
        raise NameError("Could not find data dictionary P")
>>>>>>> REPLACE

This change makes the discovery much more robust by scanning for the *structure* of the data (`dict` with `edge_order`) rather than relying on specific variable names like `P`, `data`, or `config`. This should resolve the `NameError` if the data is present in the environment under any reasonable name or location.

Additionally, I will add a safety check for the `n` value. If `n` is not in `P`, we can infer it from the vertices in `edge_order`.

